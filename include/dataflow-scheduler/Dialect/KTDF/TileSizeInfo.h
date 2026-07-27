//===-- TileSizeInfo.h -------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TILESIZEINFO_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TILESIZEINFO_H_

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace scheduler {

struct AssociatedLoopInfo {
  mlir::scf::ForOp loop;
  int64_t total_size;
};

struct TileSizeInfo {
  mlir::ktdf::TilingReserveSizeOp reserve_size_op;
  llvm::SmallVector<AssociatedLoopInfo> associated_loops;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TILESIZEINFO_H_
