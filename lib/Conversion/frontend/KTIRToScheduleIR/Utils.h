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

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "ktir/Dialect/KTDP/KTDP.h"

namespace scheduler {

[[nodiscard]] inline auto getSingleUse(mlir::Value value) -> mlir::OpOperand* {
  const auto uses = value.getUses();
  if (uses.empty() || std::next(uses.begin()) != uses.end()) {
    return nullptr;
  }
  return &*uses.begin();
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
