#include "moy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::moy;

#include "moy/Dialect.cpp.inc"

struct MoyInlinerInterface: public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    /// All call operations within moy can be inlined
    bool isLegalToInline(Operation *call, Operation *callable,
                         bool wouldBeCloned) const final {
        return true;
    }

    /// All operations within moy can be inlined.
    bool isLegalToInline(Operation *, Region *, bool,
                         BlockAndValueMapping &) const final {
        return true;
    }

    /// Handle the given inlined terminator (moy.return) by replacing it with a new
    /// operation as necessary.
    void handleTerminator(Operation *op,
                          llvm::ArrayRef<Value> valuesToRepl) const final {
        // Only "moy.return" needs to be handled here.
        auto returnOp = llvm::cast<ReturnOp>(op);

        // Replace the values directly with the return operation.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it: llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

    /// Attempts to materialize a conversion for a type mismatch between a call
    /// from this dialect, and a callable region. This method should generate an
    /// operation that takes 'input' as the only operand, and produces a single
    /// result of 'resultType'. If a conversion can not be generated, nullptr
    /// should be returned.
    Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                         Type resultType,
                                         Location conversionLoc) const final {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void MoyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "moy/Ops.cpp.inc"
        >();
    addInterfaces<MoyInlinerInterface>();
}

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
    SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
    llvm::SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation.
    if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                   result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and results.
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << op->getName() << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // If all the types are the same, print the type directly.
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](Type type) { return type == resultType; })) {
        printer << resultType;
        return;
    }

    // Otherwise, print a functional type.
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, double value) {
    auto dataType = RankedTensorType::get({}, odsBuilder.getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);
    ConstantOp::build(odsBuilder, odsState, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
    mlir::DenseElementsAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
        return failure();

    result.addTypes(value.getType());
    return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
      printer << "toy.constant ";
      printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
      printer << op.value();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
    if (!resultType)
        return success();

    // Check that the rank of the attribute type matches the rank of the constant
    // result type.
    auto attrType = op.value().getType().cast<mlir::TensorType>();
    if (attrType.getRank() != resultType.getRank()) {
        return op.emitOpError("return type must match the one of the attached value "
                              "attribute: ") << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return op.emitOpError("return type shape mismatches its attribute at dimension ")
                << dim << ": " << attrType.getShape()[dim]
                << " != " << resultType.getShape()[dim];
        }
    }
    return mlir::success();
}

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.
void AddOp::inferShapes() {
    getResult().setType(getOperand(0).getType());
}

/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void CastOp::inferShapes() {
    getResult().setType(getOperand().getType());
}

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    // The inputs must be tensors with the same element type.
    TensorType input = inputs.front().dyn_cast<TensorType>();
    TensorType output = outputs.front().dyn_cast<TensorType>();
    if (!input || !output || input.getElementType() != output.getElementType())
        return false;
    // The shape is required to match if both types are ranked.
    return !input.hasRank() || !output.hasRank() || input == output;
}

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
    // Generic call always returns an unranked Tensor initially.
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() {
    return inputs();
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void MulOp::inferShapes() {
    getResult().setType(getOperand(0).getType());
}

static mlir::LogicalResult verify(ReturnOp op) {
    // We know that the parent operation is a function, because of the 'HasParent'
    // trait attached to the operation definition.
    auto function = cast<FuncOp>(op->getParentOp());

    // ReturnOps can only have a single optional operand.
    if (op.getNumOperands() > 1)
        return op.emitOpError() << "expects at most 1 return operand";

    // The operand number and types must match the function signature.
    const auto &results = function.getType().getResults();
    if (op.getNumOperands() != results.size())
        return op.emitOpError()
            << "does not return the same number of values ("
            << op.getNumOperands() << ") as the enclosing function ("
            << results.size() << ")";

    // If the operation does not have an input, we are done.
    if (!op.hasOperand())
        return mlir::success();

    auto inputType = *op.operand_type_begin();
    auto resultType = results.front();

    // Check that the result type of the function matches the operand type.
    if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
        resultType.isa<mlir::UnrankedTensorType>())
        return mlir::success();

    return op.emitError() << "type of return operand (" << inputType
                          << ") doesn't match function result type ("
                          << resultType << ")";
}

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(value);
}

void TransposeOp::inferShapes() {
    auto arrayTy = getOperand().getType().cast<RankedTensorType>();
    SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
    getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

static mlir::LogicalResult verify(TransposeOp op) {
    auto inputType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getType().dyn_cast<RankedTensorType>();
    if (!inputType || !resultType)
        return mlir::success();

    auto inputShape = inputType.getShape();
    if (!std::equal(inputShape.begin(), inputShape.end(), resultType.getShape().rbegin())) {
        return op.emitError() << "expected result shape to be a transpose of the input";
    }
    return mlir::success();
}

#define GET_OP_CLASSES
#include "moy/Ops.cpp.inc"
