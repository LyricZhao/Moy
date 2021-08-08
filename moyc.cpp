#include "moy/Dialect.h"
#include "moy/MLIRGen.h"
#include "moy/Parser.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"

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

std::unique_ptr<moy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    AST parser(lexer);
    return parser.parseModule();
}

int dumpMLIR() {
    mlir::MLIRContext context;
    // Load out dialect in this MLIR context.
    context.getOrLoadDialect<mlir::moy::MoyDialect>();

    // Handle '.moy' input to the compiler
    if (inputType != InputType::MLIR && !llvm::StringRef(inputFilename).endswith(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return 6;
        mlir::OwningModuleRef module = mlirGen(context, *moduleAST);
        if (!module)
            return 1;

        module->dump();
        return 0;
    }

    // Otherwise, the input is '.mlir'
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    // Parse the input MLIR file.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningModuleRef module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
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
