#pragma once

#include "Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace moy {

/// A variable type with shape information.
struct VarType {
    std::vector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
    enum ExprASTKind {
        Expr_VarDecl,
        Expr_Return,
        Expr_Num,
        Expr_Literal,
        Expr_Var,
        Expr_BinOp,
        Expr_Call,
        Expr_Print,
        };

    ExprAST(ExprASTKind kind, Location location): kind(kind), location(location) {}
    virtual ~ExprAST() = default;

    ExprASTKind getKind() const { return kind; }

    const Location &loc() { return location; }

private:
    const ExprASTKind kind;
    Location location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
    double Val;

public:
    NumberExprAST(Location loc, double val) : ExprAST(Expr_Num, loc), Val(val) {}

    double getValue() { return Val; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> values;
    std::vector<int64_t> dims;

public:
    LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                   std::vector<int64_t> dims)
                   : ExprAST(Expr_Literal, loc), values(std::move(values)),
                   dims(std::move(dims)) {}

                   llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
                   llvm::ArrayRef<int64_t> getDims() { return dims; }

                   /// LLVM style RTTI
                   static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

}
