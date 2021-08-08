#include "moy/MLIRGen.h"
#include "moy/AST.h"
#include "moy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>

using namespace mlir::moy;
using namespace moy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context): builder(&context) {}

    /// Public API: convert the AST for a Toy module (source file) to an MLIR
    /// Module operation.
    mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (FunctionAST &f: moduleAST) {
            auto func = mlirGen(f);
            if (!func)
                return nullptr;
            theModule.push_back(func);
        }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the Moy operations.
        if (mlir::failed(mlir::verify(theModule))) {
            theModule.emitError("module verification error");
            return nullptr;
        }

        return theModule;
    }

private:
    /// A "module" matches a Toy source file: containing a list of functions.
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    /// The symbol table maps a variable name to a value in the current scope.
    /// Entering a function creates a new scope, and the function arguments are
    /// added to the mapping. When the processing of a function is terminated, the
    /// scope is destroyed and the mappings created in this scope are dropped.
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

    /// Helper conversion for a Toy AST location to an MLIR location.
    mlir::Location loc(Location loc) {
        return mlir::FileLineColLoc::get(builder.getIdentifier(*loc.file), loc.line, loc.col);
    }

    /// Declare a variable in the current scope, return success if the variable
    /// wasn't declared yet.
    mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
        if (symbolTable.count(var))
            return mlir::failure();
        symbolTable.insert(var, value);
        return mlir::success();
    }
};

}
