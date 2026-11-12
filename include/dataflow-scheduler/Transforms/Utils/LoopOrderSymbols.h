//===-------------------------------------------------------------*- c++ -*-==//
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

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_LOOPORDERSYMBOLS_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_LOOPORDERSYMBOLS_H_

#include <llvm/ADT/StringRef.h>

namespace scheduler {

/// Discardable attribute set by symbolize-loop-order-and-data-transfer on the
/// ktdf.tiling.reserve_size op that stands for the original loop index placed
/// at depth d of a candidate set (positions[d]). Its value is d, counted from
/// the outermost loop of the set. Read by optimize-loop-order-and-tiling to
/// tell these symbols apart from tile size symbols.
inline constexpr llvm::StringLiteral kLoopOrderDepthAttr =
    "scheduler.loop_order_depth";

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_LOOPORDERSYMBOLS_H_
