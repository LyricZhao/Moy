#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "moy/ShapeInferenceInterface.h"

/// Include the auto-generated header file containing the declaration of the moy
/// dialect.
#include "moy/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// moy operations.
#define GET_OP_CLASSES
#include "moy/Ops.h.inc"
