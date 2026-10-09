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
//
// Utility functions for transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_UTILS_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_UTILS_H_

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

namespace scheduler {

/// Returns the static trip count of `loop` iff lb, ub, step are all
/// arith.constant index ops. step must be positive.
std::optional<int64_t> getStaticTripCount(mlir::scf::ForOp loop);

/// Extracts and validate grid size from function attributes
/// Grid is a 1D array attribute containing a single integer
mlir::LogicalResult extractGridSize(mlir::func::FuncOp func, int& grid_size);

/// Clones an scf.for operation and add additional iter args (and corresponding
/// return values). The initial values of the new loop-carried arguments are
/// dummy constants (index 0). The return values are simply the loop-carried
/// arguments passed through. The reason for cloning is that MLIR infrastructure
/// doesn't allow adding new results to an operation without cloning it.
///
/// If \p delete_op is set, \p loop_op will be deleted after the new loop is
/// created.
///
/// After cloning the loop body of \p loop_op, \p ir_map will contain the IR
/// mapping of the loop body. If \p delete_op was false, this IRMapping can be
/// used outside of this function.
///
/// \param loop_op The loop to update.
/// \param n_values How many additional return values/iter_args should the new
/// loop contain.
/// \param ir_map The IRMapping object to store the cloned loop body in. Only
/// valid for use if \p delete_op was false.
/// \param delete_op If true, deletes \p loop_op after creation of the new loop.
/// \return A new loop created from \p loop_op but with \p n_values number of
/// additional return values.
mlir::scf::ForOp createForOpWithAdditionalIterArgs(mlir::scf::ForOp loop_op,
                                                   int n_values,
                                                   mlir::IRMapping& ir_map,
                                                   bool delete_op = true);

/// Erases the ops of \p roots that nothing reads, and whatever they were the
/// last reader of.
void eraseDeadAncestorOps(llvm::ArrayRef<mlir::Operation*> roots);

/// Number of vector lanes the compute resource provides for `elem_type`.
int64_t getVectorLanes(mlir::Type elem_type,
                       mlir::ktdf_arch::ExecutionUnitOp compute);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_UTILS_H_

// Made with Bob
