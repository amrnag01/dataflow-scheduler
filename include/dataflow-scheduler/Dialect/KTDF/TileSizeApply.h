//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TILESIIZEAPPLY_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TILESIIZEAPPLY_H_

#include <cstdint>

#include "dataflow-scheduler/Dialect/KTDF/TileSizeInfo.h"
#include "mlir/IR/Builders.h"

namespace scheduler {

// Apply a tile size to a TileSizeInfo, replacing its reserve_size op with a constant
void applyTileSize(mlir::OpBuilder& builder, TileSizeInfo& ts_info, int64_t tile_size);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TILESIIZEAPPLY_H_
