#include "moy/Dialect.h"
#include "moy/MLIRGen.h"
#include "moy/Parser.h"
#include "moy/Passes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace moy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input moy file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

namespace {
enum InputType { Moy, MLIR };
}

static cl::opt<enum InputType> inputType(
    "x", cl::init(Moy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Moy, "moy", "load the input file as a Moy source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as a MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR };
}

static cl::opt<enum Action> emitAction("emit",
        cl::desc("Select the kind of output desired"),
        cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
        cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

std::unique_ptr<moy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    return parser.parseModule();
}

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningModuleRef &module) {
    // Handle '.moy' input to the compiler
    if (inputType != InputType::MLIR && !llvm::StringRef(inputFilename).endswith(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return 6;
        module = mlirGen(context, *moduleAST);
        return !module ? 1 : 0;
    }

    // Otherwise, the input is '.mlir'
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    // Parse the input MLIR file.
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int dumpMLIR() {
    mlir::MLIRContext context;
    // Load out dialect in this MLIR context.
    context.getOrLoadDialect<mlir::moy::MoyDialect>();

    mlir::OwningModuleRef module;
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrDiagnosticHandler(sourceMgr, &context);
    if (int error = loadMLIR(sourceMgr, context, module))
        return error;

    if (enableOpt) {
        mlir::PassManager pm(&context);
        // Apply any generic pass manager command line options and run the pipeline.
        mlir::applyPassManagerCLOptions(pm);

        // Inline all functions into main and then delete them.
        pm.addPass(mlir::createInlinerPass());

        // Now that there is only one function, we can infer the shapes of each of
        // the operations.
        mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
        optPM.addPass(mlir::moy::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());

        if (mlir::failed(pm.run(*module)))
            return 4;
    }

    module->dump();
    return 0;
}

int dumpAST() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Moy AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;

    dump(*moduleAST);
    return 0;
}

int main(int argc, char **argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "Moy compiler\n");

    switch (emitAction) {
        case Action::DumpAST:
            return dumpAST();
        case Action::DumpMLIR:
            return dumpMLIR();
        default:
            llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    return 0;
}
