#include "moy/Parser.h"

#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
    enum Action { None, DumpAST };
}

static cl::opt<enum Action> emitAction("emit",
                                       cl::desc("Select the kind of output desired"),
                                       cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "Moy compiler\n");



    return 0;
}
