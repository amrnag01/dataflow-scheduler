//===-- TileSizeAgentInterface.cpp ---*- c++ -*-===//
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

#include "dataflow-scheduler/Utils/TileSizeAgentInterface.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/Support/DebugLog.h"

namespace scheduler {

bool validateTileSizeResult(
    int64_t selected_tile_size,
    const TileSizeInfo& tile_size_info,
    int64_t min_value,
    int64_t divisibility) {

  // Rule 1: >= min_value
  if (selected_tile_size < min_value) {
    LDBG(1) << "Validation failed: tile_size " << selected_tile_size
            << " < min_value " << min_value;
    return false;
  }

  // Rule 2: divisibility constraint
  if (divisibility > 1 && selected_tile_size % divisibility != 0) {
    LDBG(1) << "Validation failed: tile_size " << selected_tile_size
            << " not divisible by " << divisibility;
    return false;
  }

  // Rule 3: loop divisibility for all loops
  for (const auto& loop_info : tile_size_info.associated_loops) {
    if (loop_info.total_size % selected_tile_size != 0) {
      LDBG(1) << "Validation failed: loop trip_count "
              << loop_info.total_size
              << " not divisible by tile_size " << selected_tile_size;
      return false;
    }
  }

  // Rule 4: positive
  if (selected_tile_size <= 0) {
    LDBG(1) << "Validation failed: tile_size " << selected_tile_size
            << " is not positive";
    return false;
  }

  return true;
}

}  // namespace scheduler
