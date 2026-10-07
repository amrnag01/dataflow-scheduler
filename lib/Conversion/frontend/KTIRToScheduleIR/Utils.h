//===-- Utils.h -------------------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_UTILS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_UTILS_H_

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "ktir/Dialect/KTDP/KTDP.h"

namespace scheduler {

using MemRef = mlir::TypedValue<mlir::MemRefType>;
using AccessTile = mlir::TypedValue<mlir::ktdp::AccessTileType>;

[[nodiscard]] inline auto getMemorySpace(MemRef memref) -> mlir::Attribute {
  if (const auto space = memref.getType().getMemorySpace(); space) {
    return space;
  }

  if (auto source = memref.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>()) {
    return source.getMemorySpace();
  }

  return nullptr;
}

[[nodiscard]] inline auto getMemorySpace(AccessTile access_tile)
    -> mlir::Attribute {
  auto source = access_tile.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
  if (!source) {
    return nullptr;
  }

  auto base = llvm::dyn_cast<MemRef>(source.getBase());
  if (!base) {
    return nullptr;
  }

  return getMemorySpace(base);
}

[[nodiscard]] inline auto getMemorySpace(mlir::Value value) -> mlir::Attribute {
  if (auto access_tile = llvm::dyn_cast<AccessTile>(value); access_tile) {
    return getMemorySpace(access_tile);
  }
  if (auto memref = llvm::dyn_cast<MemRef>(value); memref) {
    return getMemorySpace(memref);
  }

  return nullptr;
}

struct AttrMapping : llvm::DenseMap<mlir::Attribute, mlir::Attribute> {
  using DenseMap::DenseMap;

  [[nodiscard]] auto map(mlir::Attribute attr) const -> mlir::Attribute {
    const auto mapped = lookup(attr);
    return mapped ? mapped : attr;
  }
};

[[nodiscard]] inline auto getSingleUse(mlir::Value value) -> mlir::OpOperand* {
  const auto uses = value.getUses();
  if (uses.empty() || std::next(uses.begin()) != uses.end()) {
    return nullptr;
  }
  return &*uses.begin();
}

[[nodiscard]] inline auto getThrottle(mlir::Operation* op) -> int64_t {
  const auto attr =
      op->getAttrOfType<mlir::ktdf_arch::I64Attr>(kThrottleAttrName);
  return attr ? attr.getValue() : std::numeric_limits<int64_t>::max();
};
[[nodiscard]] inline auto getThrottle(mlir::OpResult result) -> int64_t {
  auto min = std::numeric_limits<int64_t>::max();
  for (auto* user : result.getUsers()) {
    min = std::min(min, getThrottle(user));
  }
  return min;
};
[[nodiscard]] inline auto getThrottle(mlir::OpOperand& operand) -> int64_t {
  auto result = llvm::dyn_cast<mlir::OpResult>(operand.get());
  return result ? getThrottle(result.getOwner())
                : std::numeric_limits<int64_t>::max();
}

inline void setThrottle(mlir::Operation* op, int64_t value) {
  op->setAttr(kThrottleAttrName,
              mlir::ktdf_arch::I64Attr::get(op->getContext(), value));
}

[[nodiscard]] inline auto getLoopType(mlir::utils::IteratorType type)
    -> mlir::ktdf::LoopType {
  switch (type) {
    case mlir::utils::IteratorType::parallel:
      return mlir::ktdf::LoopType::ParallelLoop;
    case mlir::utils::IteratorType::reduction:
      return mlir::ktdf::LoopType::ReductionLoop;
  }
  llvm_unreachable("unknown iterator type");
}

[[nodiscard]] inline auto getLoopTypeAttr(mlir::MLIRContext* context,
                                          mlir::utils::IteratorType type)
    -> mlir::ktdf::LoopTypeAttr {
  return mlir::ktdf::LoopTypeAttr::get(context, getLoopType(type));
}

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_UTILS_H_
