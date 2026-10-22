//===-- Reuse.h -------------------------------------------------*- c++ -*-===//
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
// Broadcast-promotion candidate search: scan a ktdf.pipeline for the first
// ktdf.data_transfer that can be legally hoisted out of an enclosing
// scf.for loop.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REUSE_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REUSE_H_

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf::reuse {

/// Find the underlying memref.alloc that defines `dest`. If `dest` is a
/// direct alloc result, return it. If `dest` is a result of a
/// ktdf.private op, walk the corresponding private_yield operand back to
/// the alloc inside the private body. Returns nullptr if no alloc can be
/// found through the chain.
auto findUnderlyingAlloc(Value dest) -> memref::AllocOp;

/// A legal hoist candidate for BroadcastPromotion.
struct Candidate {
  DataTransferOp transfer;
  StageOp donor_stage;
  /// Outermost scf.for that the transfer can be hoisted past. The sibling
  /// pipeline will be inserted into target_loop's parent block,
  /// immediately before target_loop.
  scf::ForOp target_loop;
};

/// Scan `pipeline` for the first legal broadcast-hoist candidate. Stages
/// are visited in source order; within a stage, data_transfer ops are
/// visited in source order; for each transfer, the outermost legal
/// target_loop is chosen.
///
/// Legality (all must hold for the chosen target_loop and every IV
/// between the innermost enclosing loop and target_loop inclusive):
///   (a) Invariance: neither source nor destination indices/sizes depend
///       on the IV.
///   (b) Destination is not a FIFO slot.
///   (c) No write to the transfer's source memref inside the loop's
///       region.
///   (d) No write (other than the transfer itself) to the transfer's
///       destination memref inside the loop's region.
///   (e) The source memref's defining op (if any) is not inside the
///       loop's region.
///
/// Returns nullopt if no legal candidate exists in `pipeline`.
auto findFirstCandidate(PipelineOp pipeline) -> std::optional<Candidate>;

}  // namespace mlir::ktdf::reuse

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REUSE_H_
