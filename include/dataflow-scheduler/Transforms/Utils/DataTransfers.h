//===-- DataTransfers.h -----------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_DATATRANSFERS_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_DATATRANSFERS_H_

namespace mlir {
class RewritePatternSet;
}  // namespace mlir

namespace scheduler {

/// Adds @p patterns that convert various ops to `ktdf.data_transfer`.
///
///  - Converts `ktdf.write_to_fifo(ktdp_lowering.load)`
///  - Converts `ktdp_lowering.store(ktdf.read_from_fifo)`
///  - Converts `ktdp_lowering.store(ktdp_lowering.load)`
void populateConvertToDataTransferPatterns(mlir::RewritePatternSet& patterns);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_DATATRANSFERS_H_
