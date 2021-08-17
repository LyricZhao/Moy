#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace moy {
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Moy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

} // namespace moy

} // namespace mlir
