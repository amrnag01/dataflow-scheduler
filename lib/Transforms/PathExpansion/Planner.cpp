//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"

#include <llvm/Support/ErrorHandling.h>

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "path-expansion-planner"

using namespace scheduler;

namespace scheduler {

namespace {

/// Helper to create an identity affine map for intermediate buffer accesses
static mlir::AffineMap createAffineMapForIntermediateBuffer(
    const PrivateResourceSpec* buffer, mlir::MLIRContext* context) {
  return mlir::AffineMap::getMultiDimIdentityMap(buffer->shape.size(), context);
}

/// Helper to validate FIFO slot index assignment
/// Checks that the assigned slot index matches the original result index
/// from the template transfer operation
void validateFifoSlotIndex(mlir::Value orig_fifo_value,
                           size_t assigned_slot_idx,
                           const PrivateResourceSpec* fifo_spec) {
  if (auto orig_private_result =
          mlir::dyn_cast<mlir::OpResult>(orig_fifo_value)) {
    unsigned orig_result_idx = orig_private_result.getResultNumber();
    auto priv_op = orig_fifo_value.getDefiningOp<mlir::ktdf::PrivateOp>();
    assert(priv_op);
    mlir::ktdf::PrivateYieldOp yield = priv_op.getYieldOp();
    assert(orig_result_idx < yield.getOperands().size());
    mlir::Value fifo_slot_value = yield.getOperands()[orig_result_idx];
    mlir::OpResult fifo_alloc_result =
        mlir::dyn_cast<mlir::OpResult>(fifo_slot_value);
    assert(fifo_alloc_result);

    assert(assigned_slot_idx == fifo_alloc_result.getResultNumber() &&
           "expected result index to match our calculated fifo slot index");
  }
  assert(assigned_slot_idx < fifo_spec->elements_per_slot.size() &&
         "Slot index should be within the allocated slots for this FIFO type");
}

/// Helper to check if a stage is an intermediate (synthetic) stage
inline bool isIntermediateStage(const StageNode* stage) {
  return stage->getOperation() == nullptr;
}

//===----------------------------------------------------------------------===//
// Debug Output Helpers
//===----------------------------------------------------------------------===//

static void debugPrintStageList(llvm::raw_ostream& os,
                                llvm::ArrayRef<StageNode*> stages,
                                const PathExpansionPlan* plan,
                                llvm::StringRef step_name) {
  os << "\n=== " << step_name << " ===\n";
  os << "Total stages: " << stages.size() << "\n";
  for (StageNode* stage : stages) {
    os << "  Stage " << stage->getStageId() << ": ";

    if (isIntermediateStage(stage))
      os << "(synthetic, unmaterialized)";
    else
      os << "(original)";

    if (plan && plan->stage_info.count(stage)) {
      const StageMaterializationInfo& info = plan->stage_info.at(stage);
      info.print(os);
    }
    os << "\n";
  }
}

static void debugPrintResourceSpecs(llvm::raw_ostream& os,
                                    const PrivateResourceFactory& factory) {
  os << "Private resources:\n";
  size_t mem_count = 0, fifo_count = 0;
  for (const auto& spec_ptr : factory.getSpecs()) {
    if (spec_ptr->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      mem_count++;
    } else if (spec_ptr->kind == PrivateResourceSpec::Kind::kFifo) {
      fifo_count++;
    }
  }
  os << "  Memory buffers: " << mem_count << "\n";
  os << "  FIFOs: " << fifo_count << "\n";
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Print Functions for Data Structures
//===----------------------------------------------------------------------===//

void PrivateResourceSpec::print(llvm::raw_ostream& os) const {
  switch (kind) {
    case Kind::kMemoryBuffer:
      os << "MemoryBuffer(resource=" << memory_resource << ", shape=[";
      llvm::interleave(shape, os, [&](int64_t d) { os << d; }, "x");
      os << "], element_type=" << element_type << ")";
      break;
    case Kind::kFifo:
      os << "Fifo(src=" << fifo_src << ", dest=" << fifo_dest << ", slots=[";
      llvm::interleave(
          elements_per_slot, os, [&](int64_t e) { os << e; }, ", ");
      os << "], element_type=" << element_type << ")";
      break;
    case Kind::kUnknown:
      os << "Unknown";
      break;
  }
}

void PrivateResourceSpec::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

void TransferMaterializationInfo::print(llvm::raw_ostream& os) const {
  os << "Transfer(" << source_resource << " -> " << dest_resource;

  if (source_private_resource) {
    os << ", priv_source=";
    source_private_resource->print(os);
  }

  if (dest_private_resource) {
    os << ", priv_dest=";
    dest_private_resource->print(os);
  }

  if (template_op) {
    os << ", template=";
    printLocation(os, template_op);
  } else {
    os << ", synthetic";
  }

  os << ")";
}

void TransferMaterializationInfo::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

void TransferMaterializationInfo::printDebug(llvm::raw_ostream& os) const {
  os << source_resource << " -> " << dest_resource;

  if (source_private_resource) {
    os << " (source: ";
    if (source_private_resource->kind == PrivateResourceSpec::Kind::kFifo) {
      os << "FIFO slot " << source_slot_index;
    } else {
      os << "buffer";
    }
    os << ")";
  }

  if (dest_private_resource) {
    os << " (dest: ";
    if (dest_private_resource->kind == PrivateResourceSpec::Kind::kFifo) {
      os << "FIFO slot " << dest_slot_index;
    } else {
      os << "buffer";
    }
    os << ")";
  }
}

void StageMaterializationInfo::print(llvm::raw_ostream& os) const {
  os << " kind=";
  switch (kind) {
    case StageMaterializationInfo::Kind::kPreserveOriginal:
      os << "PreserveOriginal";
      break;
    case StageMaterializationInfo::Kind::kAdaptFifoKinds:
      os << "AdaptFifoKinds";
      break;
    case StageMaterializationInfo::Kind::kAdaptTransfer:
      os << "AdaptTransfer";
      break;
    case StageMaterializationInfo::Kind::kSyntheticTransfer:
      os << "SyntheticTransfer";
      break;
  }
  os << ", applicable_unit:";
  if (applicable_unit.has_value()) {
    if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(*applicable_unit)) {
      os << str_attr.getValue();
    } else {
      os << "<unknown>";
    }
  } else {
    os << "none";
  }
  os << ", transfers=" << transfers.size();

#if 1
  os << "\nTransfers:\n";
  for (const TransferMaterializationInfo* ti : transfers) {
    ti->print(os);
    os << "\n";
  }
#endif
}
void StageMaterializationInfo::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

//===----------------------------------------------------------------------===//
// Stage DAG Validation Functions
//===----------------------------------------------------------------------===//
mlir::LogicalResult validateLinearChain(
    llvm::ArrayRef<StageNode*> sorted_stages) {
  // Empty is trivially linear
  if (sorted_stages.empty()) return mlir::success();

  // Walk in topological order (sources first).
  // getDependencies() returns outgoing edges (successors), so iterating deps
  // for each stage and incrementing predecessor_count[dep] correctly builds
  // the in-degree map by the time each stage is visited.
  llvm::DenseMap<StageNode*, int> predecessor_count;
  int source_count = 0;
  int sink_count = 0;

  for (StageNode* stage : sorted_stages) {
    const llvm::SmallVector<StageNode*>& deps = stage->getDependencies();

    // fan-out check: a linear chain has at most one successor per stage.
    if (deps.size() > 1) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " has multiple outgoing dependencies (branching)";
      return mlir::failure();
    }

    // Propagate predecessor count to successors before reading our own count.
    for (StageNode* dep : deps) predecessor_count[dep]++;

    // By this point all predecessors of `stage` have been visited (topological
    // order), so predecessor_count[stage] is fully populated.
    if (predecessor_count[stage] == 0) source_count++;
    if (deps.empty()) sink_count++;

    // fan-in check: a linear chain has at most one predecessor per stage.
    if (predecessor_count[stage] > 1) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " has multiple incoming dependencies (merging)";
      return mlir::failure();
    }
  }

  if (source_count != 1 || sink_count != 1) {
    LDBG(1) << "Expected 1 source and 1 sink, got " << source_count
            << " sources and " << sink_count << " sinks";
    return mlir::failure();
  }

  return mlir::success();
}
/// Return the memory-space attribute from val's type if it is a MemRefType
/// with a memory space, otherwise return nullptr.
static ResourceType memrefMemorySpace(mlir::Value val) {
  if (auto mt = mlir::dyn_cast<mlir::MemRefType>(val.getType()))
    return mt.getMemorySpace();
  return nullptr;
}

/// Return the {source, dest} memref memory-spaces from the first
/// data_transfer op in stage_op.  Either element may be nullptr if that side
/// is not a memref (e.g. a FIFO).  Both are found in a single walk.
static std::pair<ResourceType, ResourceType> firstTransferMemSpaces(
    mlir::ktdf::StageOp stage_op) {
  ResourceType src, dst;
  stage_op.walk([&](mlir::ktdf::DataTransferOp transfer_op) {
    if (!src) src = memrefMemorySpace(transfer_op.getSource());
    if (!dst) dst = memrefMemorySpace(transfer_op.getDestination());
    return (src && dst) ? mlir::WalkResult::interrupt()
                        : mlir::WalkResult::advance();
  });
  return {src, dst};
}

//===----------------------------------------------------------------------===//
// Main Planning Functions
//===----------------------------------------------------------------------===//

/// Topologically sort pipeline stages and verify the result is a linear chain.
static llvm::FailureOr<llvm::SmallVector<StageNode*>>
sortAndValidateLinearPipelineStages(PipelineTree& tree,
                                    PipelineNode* pipeline) {
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(sorted_stages_or)) {
    LDBG(1) << "Failed to perform topological sort (cycle detected)";
    return mlir::failure();
  }

  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  if (mlir::failed(validateLinearChain(sorted_stages))) {
    LDBG(1) << "Stage topology is not a linear chain";
    return mlir::failure();
  }

  return sorted_stages;
}

/// Build full shortest path by walking consecutive pairs of original-stage
/// resources and concatenating the BFS result for each segment.
/// This correctly handles round-trip pipelines (e.g. DDR→SFU→DDR) where
/// start == end, which a single findShortestPath(DDR, DDR) call cannot resolve.
static std::optional<scheduler::arch_view::RoutingGraph::Path>
buildFullShortestPath(llvm::ArrayRef<ResourceType> original_resource_path,
                      const scheduler::arch_view::RoutingGraph& arch_graph) {
  scheduler::arch_view::RoutingGraph::Path full_path;

  for (size_t i = 0; i + 1 < original_resource_path.size(); ++i) {
    scheduler::arch_view::RoutingGraph::NodeId src =
        arch_graph.getNodeIdForResource(original_resource_path[i]);
    scheduler::arch_view::RoutingGraph::NodeId dst =
        arch_graph.getNodeIdForResource(original_resource_path[i + 1]);

    std::optional<scheduler::arch_view::RoutingGraph::Path> seg =
        arch_graph.findShortestPath(src, dst);
    if (!seg) return std::nullopt;

    // Skip the first node of seg if it duplicates the tail of the path built
    // so far (i.e. the junction node shared between consecutive segments).
    size_t start = (!full_path.empty() && !seg->empty() &&
                    full_path.back() == seg->front())
                       ? 1
                       : 0;
    full_path.append(seg->begin() + start, seg->end());
  }

  return full_path;
}

/// Assign stage_resource to every original stage using a context-aware
/// two-pass algorithm that accounts for neighboring stages.
///
/// The stage_resource is the memory endpoint adjacent to the stage's LS unit
/// in the routing graph.  It cannot always be determined from a single stage
/// in isolation when a transfer has two memref operands; the neighbor context
/// disambiguates which side is the external endpoint.
///
/// Rules (applied in order, first match wins):
///  1. Anchor stage (applicable_unit present in input IR): stage_resource ==
///     that applicable-unit resource.
///  2. First stage in chain (no left neighbor): use source memref memory-space.
///  3. Last stage in chain (no right neighbor): use dest memref memory-space.
///  4. Intermediate stage with exactly one memref side: the memref side is
///     unambiguously the stage resource — no neighbor context needed.
///  5. Intermediate stage: examine already-resolved neighbor resources —
///       a. If left neighbor's stage_resource matches this stage's transfer
///          source memref → use this stage's transfer dest memref memory-space.
///       b. Else if right neighbor's stage_resource matches this stage's
///          transfer dest memref → use this stage's transfer source memref
///          memory-space.
///       c. Otherwise assert (unresolvable without further passes).
///
/// Pass 1 resolves rules 1–4.  Pass 2 resolves rule 5 for any remaining
/// stages, iterating until stable (handles chains of intermediate stages).
static void assignOriginalStageResources(
    llvm::ArrayRef<StageNode*> sorted_stages, PathExpansionPlan* plan) {
  size_t n = sorted_stages.size();

  // Pass 1: resolve rules 1, 2, 3, 4.
  // Note: stage_info entries are default-constructed on first access (via
  // operator[])
  for (size_t i = 0; i < n; ++i) {
    StageNode* stage = sorted_stages[i];
    StageMaterializationInfo& info = plan->stage_info[stage];

    auto stage_op =
        mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
    assert(stage_op && "expected operation to be valid for original stages");

    // Rule 1: anchor stage — applicable_unit already known from input IR.
    if (auto units = stage_op.getApplicableUnits()) {
      assert(units->size() == 1 &&
             "path expansion currently does not handle nested pipelines with "
             "multi-unit stages");
      info.stage_resource = units->getValue()[0];
      continue;
    }

    auto [src_ms, dst_ms] = firstTransferMemSpaces(stage_op);

    // Rule 2: first stage — use source memref.
    if (i == 0) {
      assert(src_ms &&
             "First stage must have a memref source on its data_transfer");
      info.stage_resource = src_ms;
      continue;
    }

    // Rule 3: last stage — use dest memref.
    if (i == n - 1) {
      assert(dst_ms &&
             "Last stage must have a memref dest on its data_transfer");
      info.stage_resource = dst_ms;
      continue;
    }

    // Rule 4: exactly one side of the transfer is a memref (the other side
    // is a FIFO or similar non-memref).  The memref side is unambiguously the
    // stage resource — no neighbor context needed.
    if (src_ms && !dst_ms) {
      info.stage_resource = src_ms;
      continue;
    }
    if (dst_ms && !src_ms) {
      info.stage_resource = dst_ms;
      continue;
    }
    // Rule 5 will be handled in pass 2.
  }

  // Pass 2: resolve rule 5 iteratively until stable.
  // Each iteration resolves at least one previously-unresolved intermediate
  // stage (using a neighbor resolved in the previous iteration).
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 1; i + 1 < n; ++i) {
      StageNode* stage = sorted_stages[i];
      StageMaterializationInfo& info = plan->stage_info[stage];
      if (info.stage_resource) continue;  // already resolved

      auto stage_op =
          mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
      assert(stage_op && "expected operation to be valid for original stages");

      auto [src_ms, dst_ms] = firstTransferMemSpaces(stage_op);

      ResourceType left_resource =
          plan->stage_info[sorted_stages[i - 1]].stage_resource;
      ResourceType right_resource =
          plan->stage_info[sorted_stages[i + 1]].stage_resource;

      // Rule 5a: left neighbor's resource matches our transfer source
      //          → our endpoint is the destination side.
      if (left_resource && src_ms && left_resource == src_ms) {
        assert(dst_ms &&
               "Intermediate stage transfer must have a dest memref "
               "when resolved via left neighbor");
        info.stage_resource = dst_ms;
        changed = true;
        continue;
      }

      // Rule 5b: right neighbor's resource matches our transfer destination
      //          → our endpoint is the source side.
      if (right_resource && dst_ms && right_resource == dst_ms) {
        assert(src_ms &&
               "Intermediate stage transfer must have a source memref "
               "when resolved via right neighbor");
        info.stage_resource = src_ms;
        changed = true;
        continue;
      }
    }
  }

  // Verify all stages were resolved.
  for (size_t i = 0; i < n; ++i) {
    assert(plan->stage_info[sorted_stages[i]].stage_resource &&
           "Unable to determine stage_resource for original stage");
  }
}

/// Step 1: Build the expanded stage list by walking the routing-graph path
/// (from left to right).
/// As each node is visited, examine the original stages to see if this node is
/// relevant to any of them (either as an applicable_unit or as a
/// stage_resource). If no corresponding original stage is found, creates
/// synthetic nodes into the pipeline and rebuild the linear dependency chain.
static llvm::FailureOr<llvm::SmallVector<StageNode*>> buildExpandedStageList(
    PipelineTree& tree, PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph::Path& path,
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  using RK = scheduler::arch_view::RoutingGraph::ResourceNode::ResourceKind;

  // Ordered list of stages in the expanded pipeline (dependency order)
  llvm::SmallVector<StageNode*> ordered_stages;
  // Tracks which original stages have already been assigned
  llvm::SmallPtrSet<StageNode*, 8> visited;
  // Tracks the last stage added, so synthetic stages are inserted after it
  // in the pipeline's sibling list (which controls materializer output order).
  StageNode* last_inserted = nullptr;

  // Pre-walk: build an ordered list of sorted-stage indices for every Compute
  // node encountered on the path, in path order.  This drives the
  // direction-aware lookup: between two consecutive compute nodes at sorted
  // indices A and B, Memory/LS lookups are restricted to sorted_stages[A+1, B).
  // Before the first compute node lookups are restricted to [0, first_anchor).
  // After the last compute node lookups are restricted to (last_anchor, end].
  llvm::SmallVector<size_t> anchor_sorted_indices;
  for (size_t pi = 0; pi < path.size(); ++pi) {
    auto pnode_opt = arch_graph.getNode(path[pi]);
    if (!pnode_opt || pnode_opt->kind != RK::Compute) continue;
    // Find the sorted-stage index for this compute resource.
    for (size_t k = 0; k < sorted_stages.size(); ++k) {
      auto it = plan->stage_info.find(sorted_stages[k]);
      if (it == plan->stage_info.end()) continue;
      if (it->second.stage_resource == pnode_opt->resource) {
        anchor_sorted_indices.push_back(k);
        break;
      }
    }
  }

  // Direction-aware lookup bounds [search_begin, search_end).
  // search_end is initialised to the first anchor's sorted index (or the end of
  // sorted_stages if there are no compute nodes) so that load-side lookups
  // never reach store-side stages.  Both bounds advance each time a Compute
  // node is processed (Case C).
  size_t next_anchor_cursor = 0;  // index into anchor_sorted_indices
  size_t search_begin = 0;
  size_t search_end = anchor_sorted_indices.empty() ? sorted_stages.size()
                                                    : anchor_sorted_indices[0];

  // Helper: append a stage to ordered_stages and update last_inserted.
  auto insertStage = [&](StageNode* stage) {
    ordered_stages.push_back(stage);
    last_inserted = stage;
  };

  auto createSyntheticStage = [&](ResourceType stage_resource,
                                  ResourceType applicable_unit) -> StageNode* {
    StageNode* synth = tree.createStageNode(nullptr, next_stage_id++);
    StageMaterializationInfo info;
    info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
    info.stage_resource = stage_resource;
    info.applicable_unit = applicable_unit;
    plan->stage_info[synth] = info;
    pipeline->insertChildNode(synth, last_inserted);
    return synth;
  };

  // Helper: find the first unvisited stage whose stage_resource == resource
  // within the direction-aware search window [search_begin, search_end).
  auto findStage = [&](ResourceType resource) -> StageNode* {
    for (size_t idx = search_begin; idx < search_end; ++idx) {
      StageNode* s = sorted_stages[idx];
      if (visited.count(s)) continue;
      auto it = plan->stage_info.find(s);
      if (it == plan->stage_info.end()) continue;
      if (it->second.stage_resource == resource) return s;
    }
    return nullptr;
  };

  auto getPathNode = [&](size_t idx) {
    auto node_opt = arch_graph.getNode(path[idx]);
    assert(node_opt && "node in path must exist in graph");
    return node_opt;
  };

  auto getRightNode = [&](size_t idx) {
    return idx + 1 < path.size() ? arch_graph.getNode(path[idx + 1])
                                 : decltype(arch_graph.getNode(path[0]))();
  };

  auto handleMemoryNode =
      [&](size_t idx,
          const scheduler::arch_view::RoutingGraph::ResourceNode& node)
      -> size_t {
    StageNode* stage = findStage(node.resource);
    auto right_opt = getRightNode(idx);
    bool right_is_ls = right_opt && right_opt->kind == RK::LoadStoreUnit;

    if (stage) {
      visited.insert(stage);
      if (right_is_ls) {
        plan->stage_info[stage].applicable_unit = right_opt->resource;
        insertStage(stage);
        return idx + 2;
      }
      insertStage(stage);
      return idx + 1;
    }

    if (right_is_ls) {
      insertStage(createSyntheticStage(node.resource, right_opt->resource));
      return idx + 2;
    }

    llvm_unreachable("unhandled scenario");
    return idx + 1;
  };

  auto handleLoadStoreNode =
      [&](size_t idx,
          const scheduler::arch_view::RoutingGraph::ResourceNode& node)
      -> size_t {
    StageNode* stage_to_use = nullptr;
    size_t next_idx = idx + 1;
    auto right_opt = getRightNode(idx);

    if (right_opt && right_opt->kind == RK::Memory) {
      StageNode* stage = findStage(right_opt->resource);
      if (stage) {
        visited.insert(stage);
        plan->stage_info[stage].applicable_unit = node.resource;
        stage_to_use = stage;
      } else {
        stage_to_use = createSyntheticStage(right_opt->resource, node.resource);
      }
      next_idx = idx + 2;
    }

    if (!stage_to_use) {
      stage_to_use = createSyntheticStage(ResourceType(), node.resource);
    }

    insertStage(stage_to_use);
    return next_idx;
  };

  auto handleComputeNode =
      [&](size_t idx,
          const scheduler::arch_view::RoutingGraph::ResourceNode& node)
      -> size_t {
    assert(next_anchor_cursor < anchor_sorted_indices.size() &&
           "Compute node in path must have a corresponding original stage");
    size_t anchor_idx = anchor_sorted_indices[next_anchor_cursor];
    ++next_anchor_cursor;
    StageNode* stage = sorted_stages[anchor_idx];
    assert(!visited.count(stage) &&
           "Compute stage should not have been visited");
    visited.insert(stage);
    plan->stage_info[stage].applicable_unit = node.resource;
    insertStage(stage);
    search_begin = anchor_idx + 1;
    search_end = (next_anchor_cursor < anchor_sorted_indices.size())
                     ? anchor_sorted_indices[next_anchor_cursor]
                     : sorted_stages.size();
    return idx + 1;
  };

  // Walk the path.
  const size_t n = path.size();
  for (size_t i = 0; i < n;) {
    auto node_opt = getPathNode(i);
    const auto& node = *node_opt;

    switch (node.kind) {
      case RK::Memory:
        i = handleMemoryNode(i, node);
        break;
      case RK::LoadStoreUnit:
        i = handleLoadStoreNode(i, node);
        break;
      case RK::Compute:
        i = handleComputeNode(i, node);
        break;
    }
  }

  // Rebuild linear dependency chain over ordered_stages.
  // First, clear all existing dependencies.
  llvm::SmallVector<StageNode*> all_pipeline_stages = pipeline->getStages();
  for (StageNode* s : all_pipeline_stages) {
    s->nullifyAllDependencies();
    s->removeNullifiedDependencies();
  }

  for (size_t k = 0; k + 1 < ordered_stages.size(); ++k) {
    ordered_stages[k]->addDependency(ordered_stages[k + 1]);
  }

  return ordered_stages;
}

//===----------------------------------------------------------------------===//
// Step 2: Classify Original Stages and Create Private Resources
//===----------------------------------------------------------------------===//

/// Step 2: Walk the expanded stage list; for each original stage that is
/// adjacent to at least one intermediate (synthetic) stage, determine its
/// materialization kind and create PrivateResourceSpec objects:
///   - Transfer stages (kAdaptTransfer): memref ↔ intermediate buffer
///   - Compute stages (kAdaptFifoKinds): read_from_fifo / write_to_fifo
static mlir::LogicalResult classifyOriginalStages(
    llvm::ArrayRef<StageNode*> expanded_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  using FifoKey = std::pair<mlir::Attribute, mlir::Attribute>;
  llvm::DenseMap<FifoKey, PrivateResourceSpec*> fifo_specs;
  llvm::DenseMap<FifoKey, size_t> fifo_next_slot;

  for (size_t i = 0; i < expanded_stages.size(); ++i) {
    StageNode* current_stage = expanded_stages[i];
    if (isIntermediateStage(current_stage)) continue;

    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    StageNode* prev_stage = (i > 0) ? expanded_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < expanded_stages.size()) ? expanded_stages[i + 1] : nullptr;

    bool prev_is_intermediate = prev_stage && isIntermediateStage(prev_stage);
    bool next_is_intermediate = next_stage && isIntermediateStage(next_stage);

    if (!prev_is_intermediate && !next_is_intermediate) continue;

    auto stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

    // --- Transfer stage (kAdaptTransfer): memref ↔ intermediate buffer ---
    stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
      mlir::Type src_type = transfer.getSource().getType();
      mlir::Type dest_type = transfer.getDestination().getType();

      bool src_is_memref = mlir::isa<mlir::MemRefType>(src_type);
      bool dest_is_memref = mlir::isa<mlir::MemRefType>(dest_type);

      // Transfer involves an intermediate buffer if exactly one side is memref
      if (src_is_memref == dest_is_memref) return mlir::WalkResult::advance();

      bool intermediate_is_source = prev_is_intermediate;
      StageMaterializationInfo& neighbor_info =
          plan->stage_info[intermediate_is_source ? prev_stage : next_stage];
      ResourceType intermediate_resource = neighbor_info.stage_resource;

      // Buffer shape from the memref tile sizes
      mlir::MemRefType memref_type =
          src_is_memref ? mlir::cast<mlir::MemRefType>(src_type)
                        : mlir::cast<mlir::MemRefType>(dest_type);

      llvm::SmallVector<mlir::OpFoldResult> tile_sizes =
          src_is_memref ? transfer.getMixedSourceSizes()
                        : transfer.getMixedDestSizes();
      llvm::SmallVector<int64_t> buffer_shape;
      for (mlir::OpFoldResult ofr : tile_sizes) {
        auto int_attr =
            mlir::dyn_cast<mlir::IntegerAttr>(ofr.dyn_cast<mlir::Attribute>());
        assert(int_attr &&
               "Expected static tile sizes for intermediate buffer");
        buffer_shape.push_back(int_attr.getInt());
      }

      PrivateResourceSpec* buffer_spec =
          plan->resource_factory.createMemoryBuffer(
              intermediate_resource, buffer_shape,
              memref_type.getElementType());

      // Use a dummy edge when no direct edge exists between resources
      // (e.g. when an LS node sits between them in the routing graph).
      scheduler::arch_view::RoutingGraph::NodeId src_node_id =
          arch_graph.getNodeIdForResource(intermediate_is_source
                                              ? intermediate_resource
                                              : stage_info.stage_resource);
      scheduler::arch_view::RoutingGraph::NodeId dst_node_id =
          arch_graph.getNodeIdForResource(intermediate_is_source
                                              ? stage_info.stage_resource
                                              : intermediate_resource);
      auto edge_opt = arch_graph.getEdgeInfo(src_node_id, dst_node_id);
      scheduler::arch_view::RoutingGraph::EdgeInfo edge{src_node_id,
                                                        dst_node_id, 1};
      if (edge_opt) {
        edge = *edge_opt;
      }

      mlir::OpBuilder builder(transfer->getContext());
      TransferMaterializationInfo* transfer_info =
          plan->transfer_factory.createFromTemplateWithBuffer(
              transfer, edge, intermediate_resource, stage_info.stage_resource,
              intermediate_is_source, buffer_spec, builder);

      stage_info.transfers.push_back(transfer_info);
      stage_info.kind = StageMaterializationInfo::Kind::kAdaptTransfer;

      return mlir::WalkResult::advance();
    });

    // --- Compute stage (kAdaptFifoKinds): read_from_fifo / write_to_fifo ---
    stage_op.walk([&](mlir::Operation* op) {
      auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op);
      auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op);
      if (!read_op && !write_op) return mlir::WalkResult::advance();

      bool is_read = (read_op != nullptr);
      mlir::Value fifo_slot =
          is_read ? read_op.getFifoSlot() : write_op.getFifoSlot();
      auto fifo_slot_type =
          mlir::cast<mlir::ktdf::FifoSlotType>(fifo_slot.getType());

      StageNode* adjacent_stage = is_read ? prev_stage : next_stage;
      bool adjacent_is_intermediate =
          is_read ? prev_is_intermediate : next_is_intermediate;

      if (!adjacent_is_intermediate) return mlir::WalkResult::advance();

      // fifo_src = applicable_unit of the adjacent LS-unit stage (load side)
      // fifo_dest = applicable_unit of the adjacent LS-unit stage (store side)
      // For read (consuming from left): src = adjacent.applicable_unit,
      //   dest = stage.stage_resource
      // For write (producing to right): src = stage.stage_resource,
      //   dest = adjacent.applicable_unit
      auto adj_it = plan->stage_info.find(adjacent_stage);
      assert(adj_it != plan->stage_info.end());
      StageMaterializationInfo& adj_info = adj_it->second;

      // For read (consuming from left): src = adjacent.applicable_unit,
      //   dest = stage.stage_resource
      // For write (producing to right): src = stage.stage_resource,
      //   dest = adjacent.applicable_unit
      ResourceType fifo_src = is_read
                                  ? adj_info.applicable_unit.value_or(nullptr)
                                  : stage_info.stage_resource;
      ResourceType fifo_dest = is_read
                                   ? stage_info.stage_resource
                                   : adj_info.applicable_unit.value_or(nullptr);
      assert(fifo_src && fifo_dest &&
             "Expected valid FIFO endpoint attributes");

      mlir::Type element_type = fifo_slot_type.getElementType();
      assert(!fifo_slot_type.isDynamicNumElements() &&
             "Dynamic FIFO sizes not supported in path expansion");
      int64_t num_elements = fifo_slot_type.getStaticNumElements();

      FifoKey fifo_key = {fifo_src, fifo_dest};
      if (fifo_specs.find(fifo_key) == fifo_specs.end()) {
        PrivateResourceSpec* spec = plan->resource_factory.createFifo(
            fifo_src, fifo_dest, {num_elements}, element_type);
        fifo_specs[fifo_key] = spec;
        fifo_next_slot[fifo_key] = 0;
      } else {
        fifo_specs[fifo_key]->elements_per_slot.push_back(num_elements);
      }

      size_t slot_idx = fifo_next_slot[fifo_key]++;
      validateFifoSlotIndex(fifo_slot, slot_idx, fifo_specs[fifo_key]);

      // Use a dummy edge since the routing graph has no direct edge between
      // stage_resource nodes (they go through LS-unit nodes).
      scheduler::arch_view::RoutingGraph::NodeId src_node_id =
          arch_graph.getNodeIdForResource(stage_info.stage_resource);
      scheduler::arch_view::RoutingGraph::NodeId adj_node_id =
          arch_graph.getNodeIdForResource(adj_info.stage_resource);
      scheduler::arch_view::RoutingGraph::EdgeInfo edge{
          is_read ? adj_node_id : src_node_id,
          is_read ? src_node_id : adj_node_id, 1};

      TransferMaterializationInfo* transfer_info =
          plan->transfer_factory.createFromFifoOp(op, edge, fifo_src, fifo_dest,
                                                  fifo_specs[fifo_key],
                                                  slot_idx, is_read);

      stage_info.transfers.push_back(transfer_info);
      stage_info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;

      LDBG(1) << "  Stage " << current_stage->getStageId() << ": "
              << (is_read ? "read_from_fifo" : "write_to_fifo")
              << " - FIFO src=" << fifo_src << ", dest=" << fifo_dest
              << ", slot " << slot_idx << ", elements=" << num_elements << "\n";

      return mlir::WalkResult::advance();
    });
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 3: Populate Synthetic Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 3: Walk intermediate stages and wire up their
/// TransferMaterializationInfo by reading adjacent stages' already-populated
/// transfer specs. Reads source/dest resources directly from adjacent stage
/// stage_resource fields (no graph edge-walk).
static mlir::LogicalResult populateIntermediateStageTransfers(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  LDBG(1) << "Populating intermediate stage transfers for "
          << sorted_stages.size() << " stages";

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    if (!isIntermediateStage(current_stage)) continue;

    assert(plan->stage_info.count(current_stage));
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    assert(prev_stage && next_stage &&
           "Intermediate stage should have both neighbors");

    ResourceType intermediate_resource = stage_info.stage_resource;

    LDBG(1) << "  Processing intermediate stage " << current_stage->getStageId()
            << " (resource: " << intermediate_resource << ")\n";

    stage_info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;

    auto prev_it = plan->stage_info.find(prev_stage);
    if (prev_it == plan->stage_info.end()) continue;
    StageMaterializationInfo& prev_info = prev_it->second;
    StageMaterializationInfo& next_info = plan->stage_info[next_stage];

    // Adjacent buffer transfers record the memory resource (stage_resource) as
    // their endpoint. Adjacent FIFO transfers record the LS unit
    // (applicable_unit). Accept either when matching transfers from neighbours.
    ResourceType ls_unit = stage_info.applicable_unit.value_or(nullptr);
    auto matchesIntermediate = [&](ResourceType r) -> bool {
      return r == intermediate_resource || (ls_unit && r == ls_unit);
    };

    // Read source/dest resources directly from adjacent stage stage_resource.
    ResourceType inferred_source_resource = prev_info.stage_resource;
    ResourceType inferred_dest_resource = next_info.stage_resource;

    llvm::SmallPtrSet<const TransferMaterializationInfo*, 4>
        visited_next_transfer;
    for (const TransferMaterializationInfo* prev_transfer :
         prev_info.transfers) {
      if (!matchesIntermediate(prev_transfer->dest_resource)) continue;

      assert(prev_transfer->dest_private_resource);
      const PrivateResourceSpec* source_resource_spec =
          prev_transfer->dest_private_resource;

      // Look for matching transfer in next stage to get destination resource
      // Find a transfer in next_stage that has intermediate_resource as source
      const PrivateResourceSpec* dest_resource_spec = nullptr;
      size_t dest_slot_idx = 0;
      llvm::SmallVector<mlir::Value> dest_indices_from_next;
      llvm::SmallVector<mlir::OpFoldResult> dest_sizes_from_next;
      mlir::AffineMap dest_map_from_next;
      for (const TransferMaterializationInfo* next_transfer :
           next_info.transfers) {
        if (visited_next_transfer.contains(next_transfer)) continue;
        if (matchesIntermediate(next_transfer->source_resource)) {
          dest_resource_spec = next_transfer->source_private_resource;
          dest_slot_idx = next_transfer->source_slot_index;
          dest_indices_from_next = next_transfer->source_indices;
          dest_sizes_from_next = next_transfer->source_sizes;
          dest_map_from_next = next_transfer->source_map;
          visited_next_transfer.insert(next_transfer);
          break;
        }
      }
      assert(dest_resource_spec);

      // Build edge: use source/dest node IDs derived from adjacent stage
      // resources
      scheduler::arch_view::RoutingGraph::NodeId src_nid =
          arch_graph.getNodeIdForResource(inferred_source_resource);
      scheduler::arch_view::RoutingGraph::NodeId dst_nid =
          arch_graph.getNodeIdForResource(inferred_dest_resource);
      scheduler::arch_view::RoutingGraph::EdgeInfo edge{src_nid, dst_nid, 1};
      // Try to find a real edge (may not exist if going through LS nodes)
      if (auto real_edge = arch_graph.getEdgeInfo(src_nid, dst_nid))
        edge = *real_edge;

      mlir::MLIRContext* ctx = prev_stage->getOperation()->getContext();
      mlir::AffineMap source_map_for_intermediate = prev_transfer->dest_map;
      if (!source_map_for_intermediate && source_resource_spec &&
          source_resource_spec->kind ==
              PrivateResourceSpec::Kind::kMemoryBuffer) {
        source_map_for_intermediate =
            createAffineMapForIntermediateBuffer(source_resource_spec, ctx);
      }

      mlir::AffineMap dest_map_for_intermediate = dest_map_from_next;
      if (!dest_map_for_intermediate && dest_resource_spec &&
          dest_resource_spec->kind ==
              PrivateResourceSpec::Kind::kMemoryBuffer) {
        dest_map_for_intermediate =
            createAffineMapForIntermediateBuffer(dest_resource_spec, ctx);
      }

      TransferMaterializationInfo* transfer_info =
          plan->transfer_factory.createSynthetic(
              edge, inferred_source_resource, inferred_dest_resource,
              source_resource_spec, prev_transfer->dest_slot_index,
              prev_transfer->dest_indices, prev_transfer->dest_sizes,
              source_map_for_intermediate, dest_resource_spec, dest_slot_idx,
              dest_indices_from_next, dest_sizes_from_next,
              dest_map_for_intermediate, ctx);

      stage_info.transfers.push_back(transfer_info);

      LDBG(1) << "    Created transfer: " << inferred_source_resource << " -> "
              << inferred_dest_resource << "\n";
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Orchestration
//===----------------------------------------------------------------------===//

static bool hasAnchorStage(llvm::ArrayRef<StageNode*> sorted_stages) {
  return llvm::any_of(sorted_stages, [](StageNode* stage) {
    auto op =
        mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
    return op && op.getApplicableUnits().has_value();
  });
}

static llvm::SmallVector<ResourceType> collectOriginalStageResourcePath(
    llvm::ArrayRef<StageNode*> sorted_stages, const PathExpansionPlan* plan) {
  llvm::SmallVector<ResourceType> endpoint_path;
  for (StageNode* stage : sorted_stages) {
    ResourceType res = plan->stage_info.at(stage).stage_resource;
    if (res) endpoint_path.push_back(res);
  }
  return endpoint_path;
}

static bool needsExpansion(
    llvm::ArrayRef<ResourceType> endpoint_path,
    const scheduler::arch_view::RoutingGraph::Path& full_path,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  llvm::SmallVector<ResourceType> full_path_no_ls;
  for (auto node_id : full_path) {
    auto node_opt = arch_graph.getNode(node_id);
    assert(node_opt);
    if (node_opt->kind != scheduler::arch_view::RoutingGraph::ResourceNode::
                              ResourceKind::LoadStoreUnit) {
      full_path_no_ls.push_back(node_opt->resource);
    }
  }
  return full_path_no_ls != endpoint_path;
}

static void debugPrintInitialPlannerState(
    PipelineNode* pipeline, llvm::ArrayRef<StageNode*> sorted_stages) {
  llvm::dbgs() << "Original PipelineTree before expansion:\n";
  pipeline->print(llvm::dbgs());
  llvm::dbgs() << "\nOriginal topological order: ";
  for (StageNode* stage : sorted_stages) {
    llvm::dbgs() << "Stage " << stage->getStageId() << " ";
  }
  llvm::dbgs() << "\n\n";
}

static void debugPrintFullPath(
    const scheduler::arch_view::RoutingGraph::Path& full_path) {
  llvm::dbgs() << "Full shortest path (node IDs): ";
  for (auto node_id : full_path) llvm::dbgs() << node_id << " ";
  llvm::dbgs() << "\n";
}

static void debugPrintPlanningSummary(
    llvm::ArrayRef<StageNode*> expanded_stages, const PathExpansionPlan* plan) {
  llvm::dbgs() << "\n=== Planning Complete ===\n";
  llvm::dbgs() << "Total stages: " << expanded_stages.size() << "\n";
  llvm::dbgs() << "Total private resources: "
               << plan->resource_factory.getSpecs().size() << "\n";
  debugPrintResourceSpecs(llvm::dbgs(), plan->resource_factory);
}

/// Run the 3-step path expansion materialization pipeline.
static mlir::LogicalResult applyPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph::Path& full_path,
    const llvm::SmallVector<StageNode*>& sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  // STEP 1: Build expanded stage list
  LDBG(1) << "\n=== Step 1: Build Expanded Stage List ===\n";
  auto expanded_stages_or =
      buildExpandedStageList(tree, pipeline, full_path, sorted_stages,
                             arch_graph, plan, next_stage_id);
  if (mlir::failed(expanded_stages_or)) {
    return mlir::failure();
  }
  llvm::SmallVector<StageNode*> expanded_stages = *expanded_stages_or;
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 1");
  });

  // STEP 2: Classify original stages and create private resources
  LDBG(1) << "\n=== Step 2: Classify Original Stages ===\n";
  if (mlir::failed(classifyOriginalStages(expanded_stages, arch_graph, plan))) {
    return mlir::failure();
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 2");
  });

  // STEP 3: Populate synthetic stage transfers
  LDBG(1) << "\n=== Step 3: Populate Synthetic Stage Transfers ===\n";
  if (mlir::failed(populateIntermediateStageTransfers(expanded_stages,
                                                      arch_graph, plan))) {
    return mlir::failure();
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 3");
  });
  LLVM_DEBUG(debugPrintPlanningSummary(expanded_stages, plan));

  return mlir::success();
}

std::unique_ptr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  auto plan = std::make_unique<PathExpansionPlan>();
  plan->changed = false;

  // PREP 1: Sort and validate pipeline stages
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      sortAndValidateLinearPipelineStages(tree, pipeline);
  if (mlir::failed(sorted_stages_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  // PREP 2: Verify pipeline has an anchor stage
  if (!hasAnchorStage(sorted_stages)) {
    return nullptr;
  }

  LLVM_DEBUG(debugPrintInitialPlannerState(pipeline, sorted_stages));

  // PREP 3: Assign resources and collect endpoint path
  assignOriginalStageResources(sorted_stages, plan.get());
  llvm::SmallVector<ResourceType> endpoint_path =
      collectOriginalStageResourcePath(sorted_stages, plan.get());
  if (endpoint_path.size() < 2) {
    // Need at least source and destination
    plan->changed = false;
    return plan;
  }

  // PREP 4: Build full shortest path across architecture graph
  std::optional<scheduler::arch_view::RoutingGraph::Path> full_path_opt =
      buildFullShortestPath(endpoint_path, arch_graph);
  if (!full_path_opt) {
    LDBG(1) << "No resource path found\n";
    return nullptr;
  }
  scheduler::arch_view::RoutingGraph::Path full_path = *full_path_opt;

  LLVM_DEBUG(debugPrintFullPath(full_path));

  // PREP 5: Check whether path expansion is needed
  if (!needsExpansion(endpoint_path, full_path, arch_graph)) {
    plan->changed = false;
    LDBG(1) << "Pipeline already legal";
    return plan;
  }

  plan->changed = true;

  int next_stage_id = static_cast<int>(sorted_stages.size());
  if (mlir::failed(applyPathExpansion(tree, pipeline, full_path, sorted_stages,
                                      arch_graph, plan.get(), next_stage_id))) {
    return nullptr;
  }

  return plan;
}

}  // namespace scheduler

// Made with Bob
