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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_DATATRANSFERS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_DATATRANSFERS_H_

namespace mlir {
class RewritePatternSet;
}  // namespace mlir

namespace scheduler {

/// Adds @p patterns that convert various ops to `ktdf.data_transfer`.
///
///  - Converts `ktdf.write_to_fifo(ktdp_lowering.load)`
///  - Converts `ktdp_lowering.store(ktdf.read_from_fifo)`
///  - Converts `ktdp_lowering.store(ktdp_lowering.load)`
///  - Converts `ktdf.write_to_fifo(ktdp_lowering.ind_load)` to
///    `ktdf.ind_data_transfer`
///  - Converts `ktdp_lowering.ind_store(ktdf.read_from_fifo)` to
///    `ktdf.ind_data_transfer`
///  - Folds reshapes at FIFO boundaries (see
///    populateFoldFifoReshapePatterns)
void populateConvertToDataTransferPatterns(mlir::RewritePatternSet& patterns);

/// Adds @p patterns that fold `tensor.(collapse|expand)_shape` ops at FIFO
/// boundaries, since a FIFO slot is flat:
///
///  - Folds `reshape(ktdf.read_from_fifo)` into the read's result type
///  - Folds `ktdf.write_to_fifo(reshape)` into a write of the reshape source
void populateFoldFifoReshapePatterns(mlir::RewritePatternSet& patterns);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_DATATRANSFERS_H_
