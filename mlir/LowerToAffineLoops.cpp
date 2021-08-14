#include "moy/Dialect.h"
#include "moy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
    assert(type.hasRank() && "expected only ranked shaped");
    return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);

    // Make sure to allocate at the beginning of the block.
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // Make sure to deallocate this alloc at the end of the block. This is fine
    // as moy functions have no control flow.
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Create a nest of affine loops, with one loop per dimension of the shape.
    // The buildAffineLoopNest function takes a callback that is used to construct
    // the body of the innermost loop given a builder, a location and a range of
    // loop induction variables.
    llvm::SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
    llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
    buildAffineLoopNest(
        rewriter, loc, lowerBounds, tensorType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            // Call the processing function with the rewriter, the memref operands,
            // and the loop induction variables. This function will return the value
            // to store at the current index.
            Value valueToStore = processIteration(nestedBuilder, operands, ivs);
            nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
        });

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
}

namespace {

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering: public ConversionPattern {
    explicit BinaryOpLowering(MLIRContext *ctx):
        ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, llvm::ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](OpBuilder &builder, ValueRange memRefOperands,
                             ValueRange loopIvs) {
                            // Generate an adaptor for the remapped operands of the BinaryOp. This
                            // allows for using the nice named accessors that are generated by the
                            // ODS.
                            typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                            // Generate loads for the element of 'lhs' and 'rhs' at the inner
                            // loop.
                            auto loadedLHS =
                                builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                            auto loadedRHS =
                                builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                            // Create the binary operation performed on the loaded values.
                            return builder.create<LoweredBinaryOp>(loc, loadedLHS, loadedRHS);
                       });
        return success();
    }
};

using AddOpLowering = BinaryOpLowering<moy::AddOp, AddFOp>;
using MulOpLowering = BinaryOpLowering<moy::MulOp, MulFOp>;

struct ConstantOpLowering: public OpRewritePattern<moy::ConstantOp> {
    using OpRewritePattern<moy::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(moy::ConstantOp op,
                                  PatternRewriter &rewriter) const final {
        DenseElementsAttr constantValue = op.value();
        Location loc = op->getLoc();

        // When lowering the constant operation, we allocate and assign the
        // values to a corresponding memref allocation.
        auto tensorType = op.getType().cast<TensorType>();
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // We will be generating constant indices up-to the largest dimension.
        // Create these constants up-front to avoid large amounts of redundant
        // operations.
        auto valueShape = memRefType.getShape();
        llvm::SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty()) {
            for (auto i: llvm::seq<int64_t>(0, *std::max_element(valueShape.begin(), valueShape.end()))) {
                constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
            }
        } else {
            // This is the case of a tensor of rank 0.
            constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        }

        // The constant operation represents a multi-dimensional constant, so we
        // will need to generate a store for each of the elements. The following
        // functor recursively walks the dimensions of the constant shape,
        // generating a store when the recursion hits the base case.
        llvm::SmallVector<Value, 2> indices;
        auto valueIt = constantValue.getValues<FloatAttr>().begin();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
            // The last dimension is the base case of recursion, at this point
            // we store the element at the given index.
            if (dimension == valueShape.size()) {
                rewriter.create<AffineStoreOp>(
                    loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
                    llvm::makeArrayRef(indices));
                return;
            }

            // Otherwise, iterate over the current dimension and add the indices to
            // the list.
            for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++ i) {
                indices.push_back(constantIndices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        // Start the element storing recursion from the first dimension.
        storeElements(0);

        // Replace this operation with the generated alloc.
        rewriter.replaceOp(op, alloc);
        return success();
    }
};

struct ReturnOpLowering: public OpRewritePattern<moy::ReturnOp> {
    using OpRewritePattern<moy::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(moy::ReturnOp op,
                                  PatternRewriter &rewriter) const final {
        // During this lowering, we expect that all function calls have been
        // inlined.
        if (op.hasOperand())
            return failure();

        // We lower "toy.return" directly to "std.return"
        rewriter.replaceOpWithNewOp<ReturnOp>(op);
        return success();
    }
};

}
