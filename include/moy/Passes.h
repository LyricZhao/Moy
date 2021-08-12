#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace moy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace moy

} // namespace mlir
