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

#include <tuple>

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
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

/// Return the memory-space attribute from val's type if it is a MemRefType
/// with a memory space, otherwise return nullptr.
static ResourceType memrefMemorySpace(mlir::Value val) {
  if (auto mt = mlir::dyn_cast<mlir::MemRefType>(val.getType()))
    return llvm::cast<ResourceType>(mt.getMemorySpace());
  return nullptr;
}

/// Return the {source, dest} memref memory-spaces from the first
/// data_transfer op in stage_op.  Either element may be nullptr if that side
/// is not a memref (e.g. a FIFO).  Both are found in a single walk.
///
/// For stages containing an IndDataTransferOp, that op takes priority:
/// - Gather (ind_src present): returns (dir_src memory-space, nullptr).
/// - Scatter (ind_dst present): returns (nullptr, dir_dst memory-space).
/// This ensures the IAB-fill DataTransferOp does not
/// pollute the result with its "IAB" destination memory-space.
static std::pair<ResourceType, ResourceType> firstTransferMemSpaces(
    mlir::ktdf::StageOp stage_op) {
  // Check for IndDataTransferOp first — it takes priority over any co-located
  // DataTransferOp (e.g. the IBR fill inside scf.if in scatter stages).
  mlir::ktdf::IndDataTransferOp ind_transfer;
  stage_op.walk([&](mlir::ktdf::IndDataTransferOp op) {
    ind_transfer = op;
    return mlir::WalkResult::interrupt();
  });
  if (ind_transfer) {
    if (ind_transfer.isGather())
      return {memrefMemorySpace(ind_transfer.getDirSrc()), nullptr};
    // scatter
    return {nullptr, memrefMemorySpace(ind_transfer.getDirDst())};
  }

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
// Stage Side Classification
//===----------------------------------------------------------------------===//

namespace {
/// Where an original stage sits relative to the compute stages of the
/// pipeline. Load-side stages bring data towards the compute stages and
/// store-side stages carry results away from them.
enum class StageSide { kLoad, kCompute, kStore };
}  // namespace

using StageSideMap = llvm::DenseMap<StageNode*, StageSide>;

/// Return true if \p stage is pinned to a compute unit in the input IR.
///
/// Only stages whose applicable unit is a Compute node in the routing graph
/// count. A stage pinned to a load/store unit is still an ordinary load-side
/// or store-side stage.
static bool isComputeAnchor(
    StageNode* stage, const scheduler::arch_view::RoutingGraph& arch_graph) {
  auto stage_op =
      mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
  if (!stage_op) return false;
  std::optional<mlir::ArrayAttr> units = stage_op.getApplicableUnits();
  if (!units || units->size() != 1) return false;
  auto unit = llvm::dyn_cast<ResourceType>(units->getValue()[0]);
  if (!unit) return false;
  auto node = arch_graph.getNode(unit);
  return node && node->kind == scheduler::arch_view::RoutingGraph::
                                   ResourceNode::ResourceKind::Compute;
}

/// Decide, for every original stage, whether it is on the load side, is a
/// compute stage, or is on the store side. Fails if the stage graph does not
/// have a shape that path expansion can handle.
///
/// The stages do not have to form a single chain. Each side may branch and
/// merge freely, for example two load stages feeding one compute stage. What
/// matters is that every stage is clearly before or clearly after the compute
/// stages:
///
///  - There must be at least one compute stage.
///  - Every other stage must be upstream of a compute stage (load side) or
///    downstream of one (store side), directly or through other stages.
///  - No stage may be both upstream and downstream of compute stages, since
///    that would place it between two of them. Compute stages themselves may
///    feed each other directly or be unconnected.
///
/// How it works: walk the stages in topological order, marking as downstream
/// every stage fed by a compute stage or by a stage already marked
/// downstream. Then walk in reverse order to mark the upstream stages the
/// same way. Every non-compute stage must end up with exactly one mark.
static llvm::FailureOr<StageSideMap> classifyStageSides(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  llvm::SmallPtrSet<StageNode*, 4> anchors;
  for (StageNode* stage : sorted_stages)
    if (isComputeAnchor(stage, arch_graph)) anchors.insert(stage);
  if (anchors.empty()) {
    LDBG(1) << "Pipeline has no stage pinned to a compute unit";
    return mlir::failure();
  }

  // Forward walk: a stage comes after a compute stage if any of its
  // predecessors is a compute stage or comes after one.
  llvm::SmallPtrSet<StageNode*, 8> after_compute;
  for (StageNode* stage : sorted_stages) {
    if (!anchors.contains(stage) && !after_compute.contains(stage)) continue;
    for (StageNode* succ : stage->getDependencies()) after_compute.insert(succ);
  }

  // Backward walk: a stage comes before a compute stage if any of its
  // successors is a compute stage or comes before one.
  llvm::SmallPtrSet<StageNode*, 8> before_compute;
  for (StageNode* stage : llvm::reverse(sorted_stages)) {
    if (llvm::any_of(stage->getDependencies(), [&](StageNode* succ) {
          return anchors.contains(succ) || before_compute.contains(succ);
        }))
      before_compute.insert(stage);
  }

  StageSideMap sides;
  for (StageNode* stage : sorted_stages) {
    if (anchors.contains(stage)) {
      sides[stage] = StageSide::kCompute;
      continue;
    }
    bool is_before = before_compute.contains(stage);
    bool is_after = after_compute.contains(stage);
    if (is_before && is_after) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " sits between two compute stages";
      return mlir::failure();
    }
    if (!is_before && !is_after) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " is not connected to any compute stage";
      return mlir::failure();
    }
    sides[stage] = is_before ? StageSide::kLoad : StageSide::kStore;
  }
  return sides;
}

//===----------------------------------------------------------------------===//
// Main Planning Functions
//===----------------------------------------------------------------------===//

/// Topologically sort the pipeline stages, check that the stage graph has a
/// shape path expansion can handle, and work out which side of the compute
/// stages each stage is on. See classifyStageSides() for the rules.
static llvm::FailureOr<llvm::SmallVector<StageNode*>> sortAndValidateStages(
    PipelineTree& tree, PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph& arch_graph, StageSideMap& sides) {
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(sorted_stages_or)) {
    LDBG(1) << "Failed to perform topological sort (cycle detected)";
    return mlir::failure();
  }

  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  llvm::FailureOr<StageSideMap> sides_or =
      classifyStageSides(sorted_stages, arch_graph);
  if (mlir::failed(sides_or)) {
    LDBG(1) << "Stages are not all clearly before or after the compute stages";
    return mlir::failure();
  }
  sides = std::move(*sides_or);

  return sorted_stages;
}

/// Work out the stage_resource of every original stage.
///
/// The stage_resource is the memory a stage's load/store unit is attached to
/// in the routing graph. Path expansion uses it to find out which memories
/// and units the data passes through between stages.
///
/// Each stage is decided on its own, using the rules below in order:
///
///  1. The stage is already pinned to a unit in the input IR (a compute unit
///     or a load/store unit). Its resource is that unit.
///  2. Only one side of the stage's data transfer is a memref; the other side
///     is a FIFO. The memref side is the stage's memory.
///  3. Both sides are memrefs, so the side of the pipeline decides:
///       - A load-side stage reads from the memory next to its load unit and
///         pushes the data towards the compute stage, so its resource is the
///         transfer's source.
///       - A store-side stage writes to the memory next to its store unit,
///         so its resource is the transfer's destination.
///
/// Because the side of a stage comes from \p sides, the rules never look at
/// neighbouring stages. That is why the order of stages within a side does
/// not matter here.
///
/// A stage with no predecessors must read from memory, because nothing in the
/// pipeline could fill a FIFO it reads from. Likewise, a stage with no
/// successors must write to memory. Both are asserted.
static void assignOriginalStageResources(
    llvm::ArrayRef<StageNode*> sorted_stages, const StageSideMap& sides,
    PathExpansionPlan* plan) {
  // Stages that some other stage feeds into.
  llvm::SmallPtrSet<StageNode*, 8> has_predecessor;
  for (StageNode* stage : sorted_stages)
    has_predecessor.insert(stage->getDependencies().begin(),
                           stage->getDependencies().end());

  // stage_info entries are default-constructed on first access (operator[]).
  for (StageNode* stage : sorted_stages) {
    StageMaterializationInfo& info = plan->stage_info[stage];

    auto stage_op =
        mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
    assert(stage_op && "expected operation to be valid for original stages");

    // Rule 1: the unit is already known from the input IR.
    if (auto units = stage_op.getApplicableUnits()) {
      assert(units->size() == 1 &&
             "path expansion currently does not handle nested pipelines with "
             "multi-unit stages");
      info.stage_resource = llvm::cast<ResourceType>(units->getValue()[0]);
      continue;
    }

    auto [src_ms, dst_ms] = firstTransferMemSpaces(stage_op);
    assert((src_ms || dst_ms) &&
           "Unable to determine stage_resource for original stage: its data "
           "transfer has no memref side");
    assert((has_predecessor.contains(stage) || src_ms) &&
           "A stage with no predecessors must have a memref source on its "
           "data_transfer");
    assert((!stage->getDependencies().empty() || dst_ms) &&
           "A stage with no successors must have a memref dest on its "
           "data_transfer");

    // Rule 2: only one side of the transfer is a memref.
    if (!src_ms || !dst_ms) {
      info.stage_resource = src_ms ? src_ms : dst_ms;
      continue;
    }

    // Rule 3: both sides are memrefs; the side of the pipeline decides.
    // Compute stages are always pinned, so they were handled by rule 1.
    StageSide side = sides.at(stage);
    assert(side != StageSide::kCompute &&
           "Compute stages should be pinned to a unit");
    info.stage_resource = side == StageSide::kLoad ? src_ms : dst_ms;
  }
}

//===----------------------------------------------------------------------===//
// Routing Paths Between Stages
//===----------------------------------------------------------------------===//

namespace {
/// The routing-graph path for one dependency between two original stages.
///
/// The path runs from the producer's resource to the consumer's resource. Its
/// first nodes belong to the producer and its last nodes to the consumer. Any
/// nodes left in the middle are hops the data passes through. Step 1 gives
/// each hop to the stage that runs on its unit, or creates a new (synthetic)
/// stage for it.
struct EdgeSegment {
  StageNode* producer = nullptr;
  StageNode* consumer = nullptr;
  scheduler::arch_view::RoutingGraph::Path path;
  // Last path index owned by the producer.
  size_t producer_end = 0;
  // First path index owned by the consumer.
  size_t consumer_begin = 0;
  // Load/store units the path gives the producer and the consumer, if any.
  ResourceType producer_unit;
  ResourceType consumer_unit;

  /// The range [middleBegin(), middleEnd()) of path indices that need new
  /// stages. It is empty when the two stages own the whole path.
  size_t middleBegin() const { return producer_end + 1; }
  size_t middleEnd() const { return consumer_begin; }
};
}  // namespace

/// Find the routing-graph path for every dependency between original stages,
/// and work out which part of each path belongs to which stage.
///
/// A stage owns the node for its own resource. A load-side stage also owns
/// the load unit right after its memory, because that unit reads the memory.
/// For the same reason a store-side stage owns the store unit right before
/// its memory. These units later become the stages' applicable units.
///
/// For example, take a load stage reading DDR that feeds a compute stage on
/// SFU, with the path DDR -> MNILU -> L1 -> L1LU -> SFU. The load stage owns
/// DDR and MNILU, the compute stage owns SFU, and L1 -> L1LU in the middle
/// is a hop that needs a stage on L1LU.
///
/// Each dependency gets its own path, so stages may branch and merge freely.
/// Fails if some dependency has no path in the routing graph.
static llvm::FailureOr<llvm::SmallVector<EdgeSegment>> buildEdgeSegments(
    llvm::ArrayRef<StageNode*> sorted_stages, const StageSideMap& sides,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    const PathExpansionPlan* plan) {
  using RK = scheduler::arch_view::RoutingGraph::ResourceNode::ResourceKind;

  auto nodeAt = [&](const EdgeSegment& seg, size_t idx) {
    auto node = arch_graph.getNode(seg.path[idx]);
    assert(node && "node in path must exist in graph");
    return *node;
  };

  llvm::SmallVector<EdgeSegment> segments;
  llvm::DenseSet<std::pair<StageNode*, StageNode*>> seen;
  for (StageNode* producer : sorted_stages) {
    for (StageNode* consumer : producer->getDependencies()) {
      if (!seen.insert({producer, consumer}).second) continue;

      EdgeSegment seg;
      seg.producer = producer;
      seg.consumer = consumer;
      std::optional<scheduler::arch_view::RoutingGraph::Path> path =
          arch_graph.findShortestPath(
              arch_graph.getNodeIdForResource(
                  plan->stage_info.at(producer).stage_resource),
              arch_graph.getNodeIdForResource(
                  plan->stage_info.at(consumer).stage_resource));
      if (!path || path->empty()) {
        LDBG(1) << "No routing path from stage " << producer->getStageId()
                << " to stage " << consumer->getStageId();
        return mlir::failure();
      }
      seg.path = std::move(*path);

      size_t last = seg.path.size() - 1;
      seg.consumer_begin = last;

      // A load-side producer owns the load unit right after its memory,
      // unless that unit is the consumer's own resource.
      if (sides.at(producer) == StageSide::kLoad && last > 1 &&
          nodeAt(seg, 0).kind == RK::Memory &&
          nodeAt(seg, 1).kind == RK::LoadStoreUnit) {
        seg.producer_end = 1;
        seg.producer_unit = nodeAt(seg, 1).resource;
      }

      // A store-side consumer owns the store unit right before its memory,
      // unless the producer already owns it.
      if (sides.at(consumer) == StageSide::kStore &&
          last > seg.producer_end + 1 && nodeAt(seg, last).kind == RK::Memory &&
          nodeAt(seg, last - 1).kind == RK::LoadStoreUnit) {
        seg.consumer_begin = last - 1;
        seg.consumer_unit = nodeAt(seg, last - 1).resource;
      }

      segments.push_back(std::move(seg));
    }
  }
  return segments;
}

/// Return true if path expansion has to add stages: some path passes through
/// a memory or compute unit that no stage owns. A load/store unit with no
/// stage does not count on its own.
static bool needsExpansion(
    llvm::ArrayRef<EdgeSegment> segments,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  for (const EdgeSegment& seg : segments) {
    for (size_t i = seg.middleBegin(); i < seg.middleEnd(); ++i) {
      auto node = arch_graph.getNode(seg.path[i]);
      assert(node && "node in path must exist in graph");
      if (node->kind != scheduler::arch_view::RoutingGraph::ResourceNode::
                            ResourceKind::LoadStoreUnit)
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Step 1: Add Stages Along Each Path
//===----------------------------------------------------------------------===//

/// For each dependency between two original stages that needed extra stages,
/// those stages in data-flow order. They are new (synthetic) stages, or
/// existing stages that already run on the unit the data must pass through.
using EdgeChainMap = llvm::DenseMap<std::pair<StageNode*, StageNode*>,
                                    llvm::SmallVector<StageNode*, 2>>;

/// Set the applicable unit of every original stage, and check that no two of
/// them run on the same unit. A pipeline can run only one stage on each unit,
/// so an input where two stages end up on the same unit is invalid.
///
/// A stage pinned to a compute or load/store unit runs on that unit. Any other
/// stage gets the load/store unit its paths give it (see buildEdgeSegments).
/// A stage with several dependencies on the same side must get the same unit
/// from all of them.
static mlir::LogicalResult assignOriginalStageUnits(
    llvm::ArrayRef<EdgeSegment> segments,
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  using RK = scheduler::arch_view::RoutingGraph::ResourceNode::ResourceKind;

  // A stage whose resource is a compute or load/store unit runs on it.
  for (StageNode* stage : sorted_stages) {
    StageMaterializationInfo& info = plan->getStageInfo(stage);
    auto node = arch_graph.getNode(info.stage_resource);
    if (node && node->kind != RK::Memory)
      info.applicable_unit = info.stage_resource;
  }

  auto setUnit = [&](StageNode* stage,
                     ResourceType unit) -> mlir::LogicalResult {
    if (!unit) return mlir::success();
    std::optional<ResourceType>& current =
        plan->getStageInfo(stage).applicable_unit;
    if (current && *current != unit)
      return stage->getOperation()->emitError(
                 "path-expansion: stage is reached through two different "
                 "units, ")
             << *current << " and " << unit;
    current = unit;
    return mlir::success();
  };
  for (const EdgeSegment& seg : segments) {
    if (mlir::failed(setUnit(seg.producer, seg.producer_unit)) ||
        mlir::failed(setUnit(seg.consumer, seg.consumer_unit)))
      return mlir::failure();
  }

  llvm::DenseMap<ResourceType, StageNode*> unit_owner;
  for (StageNode* stage : sorted_stages) {
    std::optional<ResourceType> unit =
        plan->getStageInfo(stage).applicable_unit;
    if (!unit) continue;
    auto [it, inserted] = unit_owner.try_emplace(*unit, stage);
    if (inserted) continue;
    mlir::InFlightDiagnostic diag =
        stage->getOperation()->emitError("path-expansion: stage runs on unit ")
        << *unit << ", which another stage of the pipeline already runs on";
    diag.attachNote(it->second->getOperation()->getLoc())
        << "other stage on unit " << *unit;
    return diag;
  }
  return mlir::success();
}

/// Step 1: Add the stages that move data through memories and units that have
/// no stage of their own on a path.
///
/// How it works, for each dependency between two original stages:
///  1. The middle of the path is split into hops. A memory followed by the
///     load unit that reads it, or a store unit followed by the memory it
///     writes, is one hop that moves data through that memory. A load/store
///     unit on its own is a hop with no memory.
///  2. Each hop is given to the stage that already runs on its unit, so that
///     no unit ever runs two stages. Only if no stage runs on the unit yet is
///     a new stage created for it. The existing stage may be an original
///     stage or a new stage made for another dependency, and it must use the
///     same memory as the hop.
///  3. The dependency producer -> consumer becomes
///     producer -> hop stages -> consumer. Dependencies that need no hops are
///     left alone.
///
/// Giving a hop to an existing stage makes that stage depend on the producer.
/// If the existing stage is upstream of the producer, that creates a cycle;
/// the caller detects it when it sorts the stages again.
///
/// New stages are placed in the pipeline right after their producer (after
/// any new stages already placed there), which keeps the printed order close
/// to the data-flow order. The hop stages of each dependency are recorded in
/// \p edge_chains.
static mlir::LogicalResult expandStageEdges(
    PipelineTree& tree, PipelineNode* pipeline,
    llvm::ArrayRef<EdgeSegment> segments,
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id, EdgeChainMap& edge_chains) {
  using RK = scheduler::arch_view::RoutingGraph::ResourceNode::ResourceKind;

  // The stage running on each unit. assignOriginalStageUnits() made sure the
  // original stages all run on different units.
  llvm::DenseMap<ResourceType, StageNode*> unit_owner;
  for (StageNode* stage : sorted_stages)
    if (std::optional<ResourceType> unit =
            plan->getStageInfo(stage).applicable_unit)
      unit_owner[*unit] = stage;

  // The stage after which the next new stage following a producer goes.
  llvm::DenseMap<StageNode*, StageNode*> insert_after;

  auto addDependency = [](StageNode* from, StageNode* to) {
    if (!llvm::is_contained(from->getDependencies(), to))
      from->addDependency(to);
  };

  for (const EdgeSegment& seg : segments) {
    llvm::SmallVector<StageNode*, 2> chain;
    StageNode*& last_inserted =
        insert_after.try_emplace(seg.producer, seg.producer).first->second;

    // Find or create the stage for the hop through memory \p memory (may be
    // null) on unit \p unit.
    auto addHop = [&](ResourceType memory,
                      ResourceType unit) -> mlir::LogicalResult {
      auto owner_it = unit_owner.find(unit);
      if (owner_it == unit_owner.end()) {
        StageNode* synth = tree.createStageNode(nullptr, next_stage_id++);
        StageMaterializationInfo info;
        info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
        info.stage_resource = memory;
        info.applicable_unit = unit;
        plan->stage_info[synth] = info;
        pipeline->insertChildNode(synth, last_inserted);
        last_inserted = synth;
        unit_owner[unit] = synth;
        chain.push_back(synth);
        return mlir::success();
      }

      StageNode* owner = owner_it->second;
      if (owner == seg.producer || owner == seg.consumer ||
          llvm::is_contained(chain, owner)) {
        LDBG(1) << "Path from stage " << seg.producer->getStageId()
                << " to stage " << seg.consumer->getStageId()
                << " passes through unit " << unit
                << " twice, but a unit can run only one stage";
        return mlir::failure();
      }
      if (plan->getStageInfo(owner).stage_resource != memory) {
        LDBG(1) << "Path from stage " << seg.producer->getStageId()
                << " to stage " << seg.consumer->getStageId() << " needs unit "
                << unit << " for memory " << memory << ", but stage "
                << owner->getStageId() << " already uses it for memory "
                << plan->getStageInfo(owner).stage_resource;
        return mlir::failure();
      }
      chain.push_back(owner);
      return mlir::success();
    };

    size_t end = seg.middleEnd();
    for (size_t i = seg.middleBegin(); i < end;) {
      auto node = arch_graph.getNode(seg.path[i]);
      assert(node && "node in path must exist in graph");
      decltype(node) next;
      if (i + 1 < end) next = arch_graph.getNode(seg.path[i + 1]);

      switch (node->kind) {
        case RK::Memory:
          // A memory with no stage must be followed by the unit that reads
          // it.
          if (!next || next->kind != RK::LoadStoreUnit) {
            LDBG(1) << "Memory " << node->resource << " between stage "
                    << seg.producer->getStageId() << " and stage "
                    << seg.consumer->getStageId()
                    << " has no load unit after it";
            return mlir::failure();
          }
          if (mlir::failed(addHop(node->resource, next->resource)))
            return mlir::failure();
          i += 2;
          break;
        case RK::LoadStoreUnit:
          if (next && next->kind == RK::Memory) {
            if (mlir::failed(addHop(next->resource, node->resource)))
              return mlir::failure();
            i += 2;
          } else {
            if (mlir::failed(addHop(ResourceType(), node->resource)))
              return mlir::failure();
            i += 1;
          }
          break;
        case RK::Compute:
          // Hop stages only move data between memories through load/store
          // units. Passing data through a compute unit would need a compute
          // stage that relays it, which path expansion does not create.
          LDBG(1) << "Path from stage " << seg.producer->getStageId()
                  << " to stage " << seg.consumer->getStageId()
                  << " passes through compute unit " << node->resource
                  << ", which has no stage";
          return mlir::failure();
      }
    }

    if (chain.empty()) continue;
    seg.producer->nullifyDependency(seg.consumer);
    seg.producer->removeNullifiedDependencies();
    addDependency(seg.producer, chain.front());
    for (size_t k = 0; k + 1 < chain.size(); ++k)
      addDependency(chain[k], chain[k + 1]);
    addDependency(chain.back(), seg.consumer);
    edge_chains[{seg.producer, seg.consumer}] = std::move(chain);
  }

  return mlir::success();
}

/// Put the pipeline's stages in an order where every stage comes after the
/// stages it depends on, changing the current order as little as possible.
///
/// Giving a hop to an existing stage can make it depend on a stage placed
/// after it. The materializer emits stages in pipeline order, so this keeps
/// the output in data-flow order. How it works: repeatedly take the first
/// stage, in the current order, whose predecessors have all been placed.
/// Fails if the dependencies form a cycle.
static mlir::LogicalResult orderStagesByDependencies(PipelineNode* pipeline) {
  llvm::SmallVector<StageNode*> stages = pipeline->getStages();
  if (stages.empty()) return mlir::success();

  llvm::DenseMap<StageNode*, unsigned> unplaced_preds;
  for (StageNode* stage : stages)
    for (StageNode* succ : stage->getDependencies()) ++unplaced_preds[succ];

  llvm::SmallVector<StageNode*> ordered;
  llvm::SmallPtrSet<StageNode*, 8> placed;
  while (ordered.size() < stages.size()) {
    auto next = llvm::find_if(stages, [&](StageNode* stage) {
      return !placed.contains(stage) && unplaced_preds.lookup(stage) == 0;
    });
    if (next == stages.end()) return mlir::failure();
    placed.insert(*next);
    ordered.push_back(*next);
    for (StageNode* succ : (*next)->getDependencies()) --unplaced_preds[succ];
  }
  if (ordered == stages) return mlir::success();

  // Re-link the stages in the new order where the first one used to be.
  OperationTreeNode* pos = stages.front()->getPrevSibling();
  for (StageNode* stage : stages) stage->unlink();
  for (StageNode* stage : ordered) {
    if (pos)
      pipeline->insertChildNode(stage, pos);
    else
      pipeline->insertAsFirstChild(stage);
    pos = stage;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Stage Graph Helpers
//===----------------------------------------------------------------------===//

namespace {
/// One end (producer or consumer) of a FIFO slot inside an original stage.
struct FifoEndpoint {
  mlir::Operation* op = nullptr;
  StageNode* stage = nullptr;
  // Whether the FIFO slot is the op's source operand (always true for
  // read_from_fifo, always false for write_to_fifo).
  bool fifo_is_source = false;
};

/// The producer and consumer of every FIFO slot used by the original stages.
struct FifoEndpoints {
  llvm::MapVector<mlir::Value, FifoEndpoint> producers;
  llvm::MapVector<mlir::Value, FifoEndpoint> consumers;
  // FIFO slots used by an indirect data transfer.
  llvm::SmallPtrSet<mlir::Value, 4> indirect_slots;
};

/// Ties a transfer created in Step 2 to the hop stage it exchanges data with
/// and to the original FIFO slot it replaces. Step 3 uses this to pair up the
/// transfer that writes into each hop stage with the one that reads from it.
struct TransferLink {
  // The stage the transfer belongs to.
  StageNode* stage = nullptr;
  // The hop stage on the other side of the transfer.
  StageNode* hop = nullptr;
  mlir::Value fifo_slot;
  // Whether the transfer writes into the hop stage (rather than reading from
  // it).
  bool writes_to_hop = false;
};
}  // namespace

/// Kept in insertion order so Step 3 creates transfers in a stable order.
using TransferLinkMap =
    llvm::MapVector<const TransferMaterializationInfo*, TransferLink>;
using StagePredecessorMap =
    llvm::DenseMap<StageNode*, llvm::SmallVector<StageNode*, 2>>;

/// Map each stage to the stages that feed it. Stages only record the stages
/// they feed, so this gives the other direction.
static StagePredecessorMap buildPredecessorMap(
    llvm::ArrayRef<StageNode*> stages) {
  StagePredecessorMap preds;
  for (StageNode* stage : stages)
    for (StageNode* succ : stage->getDependencies())
      preds[succ].push_back(stage);
  return preds;
}

/// Find the op that writes and the op that reads each FIFO slot used by the
/// original stages. Fails if a slot has more than one writer or reader.
static mlir::LogicalResult collectFifoEndpoints(
    llvm::ArrayRef<StageNode*> stages, FifoEndpoints& endpoints) {
  for (StageNode* stage : stages) {
    if (isIntermediateStage(stage)) continue;
    auto stage_op = mlir::cast<mlir::ktdf::StageOp>(stage->getOperation());

    auto record = [&](llvm::MapVector<mlir::Value, FifoEndpoint>& ends,
                      mlir::Value fifo_slot, mlir::Operation* op,
                      bool fifo_is_source) -> mlir::WalkResult {
      if (!ends.insert({fifo_slot, {op, stage, fifo_is_source}}).second) {
        op->emitError("path-expansion: FIFO slot has more than one ")
            << (&ends == &endpoints.producers ? "producer" : "consumer");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    };

    mlir::WalkResult result = stage_op.walk([&](mlir::Operation* op) {
      if (auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op))
        return record(endpoints.consumers, read_op.getFifoSlot(), op, true);
      if (auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op))
        return record(endpoints.producers, write_op.getFifoSlot(), op, false);
      if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
        if (mlir::isa<mlir::ktdf::FifoSlotType>(transfer.getSource().getType()))
          if (record(endpoints.consumers, transfer.getSource(), op, true)
                  .wasInterrupted())
            return mlir::WalkResult::interrupt();
        if (mlir::isa<mlir::ktdf::FifoSlotType>(
                transfer.getDestination().getType()))
          return record(endpoints.producers, transfer.getDestination(), op,
                        false);
        return mlir::WalkResult::advance();
      }
      if (auto ind_transfer =
              mlir::dyn_cast<mlir::ktdf::IndDataTransferOp>(op)) {
        bool is_gather = ind_transfer.isGather();
        mlir::Value fifo_side =
            is_gather ? ind_transfer.getDirDst() : ind_transfer.getDirSrc();
        if (!mlir::isa<mlir::ktdf::FifoSlotType>(fifo_side.getType()))
          return mlir::WalkResult::advance();
        endpoints.indirect_slots.insert(fifo_side);
        return is_gather ? record(endpoints.producers, fifo_side, op, false)
                         : record(endpoints.consumers, fifo_side, op, true);
      }
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) return mlir::failure();
  }
  return mlir::success();
}

/// Return the fifo.allocate result that \p fifo_slot (a ktdf.private result)
/// is yielded from, or a null result if it cannot be traced.
static mlir::OpResult getFifoAllocateResult(mlir::Value fifo_slot) {
  auto private_result = mlir::dyn_cast<mlir::OpResult>(fifo_slot);
  if (!private_result) return {};
  auto priv_op =
      mlir::dyn_cast<mlir::ktdf::PrivateOp>(private_result.getOwner());
  if (!priv_op) return {};
  mlir::Value yielded =
      priv_op.getYieldOp().getOperands()[private_result.getResultNumber()];
  auto alloc_result = mlir::dyn_cast<mlir::OpResult>(yielded);
  if (!alloc_result ||
      !mlir::isa<mlir::ktdf::FifoAllocateOp>(alloc_result.getOwner()))
    return {};
  return alloc_result;
}

/// Return the hop stage that original stage \p stage exchanges data with
/// over \p fifo_slot, or nullptr if it still exchanges data directly with
/// another original stage.
///
/// The FIFO's other end is normally another original stage. If Step 1 put hop
/// stages on the dependency between the two, the answer is the hop stage
/// right next to \p stage. If the other end is not in this pipeline, fall back
/// to the only stage feeding \p stage (when it reads) or fed by it (when it
/// writes), as long as that is a new stage.
static StageNode* findFifoHop(StageNode* stage, mlir::Value fifo_slot,
                              bool stage_reads, const FifoEndpoints& endpoints,
                              const EdgeChainMap& edge_chains,
                              const StagePredecessorMap& preds) {
  const llvm::MapVector<mlir::Value, FifoEndpoint>& other_ends =
      stage_reads ? endpoints.producers : endpoints.consumers;
  auto other_it = other_ends.find(fifo_slot);
  if (other_it != other_ends.end()) {
    StageNode* other = other_it->second.stage;
    auto chain_it =
        edge_chains.find(stage_reads ? std::make_pair(other, stage)
                                     : std::make_pair(stage, other));
    if (chain_it == edge_chains.end()) return nullptr;
    return stage_reads ? chain_it->second.back() : chain_it->second.front();
  }

  llvm::SmallVector<StageNode*, 2> neighbours;
  if (stage_reads)
    neighbours = preds.lookup(stage);
  else
    neighbours.append(stage->getDependencies().begin(),
                      stage->getDependencies().end());
  if (neighbours.size() == 1 && isIntermediateStage(neighbours.front()))
    return neighbours.front();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Step 2: Classify Original Stages and Create Private Resources
//===----------------------------------------------------------------------===//

/// Step 2: Adapt the original stages that exchange data with a hop stage, and
/// create the private resources (buffers and FIFOs) they need.
///
/// A hop stage sits on a dependency between two original stages and moves the
/// data through a memory or unit the path needs, such as L1. It is either a
/// new stage or an existing stage that took on the extra work (see
/// expandStageEdges). The original stages on either side must now exchange
/// data with the hop stage instead of with each other. Each data transfer,
/// read_from_fifo and write_to_fifo in an original stage is checked: its FIFO
/// slot leads to the stage at the other end of the FIFO, and if there is a hop
/// stage on the way there, the op is adapted to exchange data with it:
///   - A data transfer between memory and a FIFO (kAdaptTransfer) uses a
///     buffer in the hop stage's memory instead of the FIFO.
///   - A read_from_fifo or write_to_fifo (kAdaptFifoKinds) uses a FIFO
///     between the hop stage's unit and this stage.
///
/// Finding the hop through the FIFO, not by position in the stage list, is
/// what lets one stage exchange data with several hop stages, for example a
/// compute stage fed by two load stages.
///
/// New FIFOs are created per original fifo.allocate, so slots that came from
/// different allocations stay in different allocations. Every adapted op is
/// recorded in \p links for Step 3.
static mlir::LogicalResult classifyOriginalStages(
    llvm::ArrayRef<StageNode*> expanded_stages,
    const StagePredecessorMap& preds, const FifoEndpoints& endpoints,
    const EdgeChainMap& edge_chains,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, TransferLinkMap& links) {
  // New FIFOs, keyed by the original fifo.allocate and the new endpoints.
  using FifoKey =
      std::tuple<mlir::Operation*, mlir::Attribute, mlir::Attribute>;
  llvm::DenseMap<FifoKey, PrivateResourceSpec*> fifo_specs;
  llvm::DenseMap<FifoKey, size_t> fifo_next_slot;

  // Only the original stages at either end of a dependency that got hop
  // stages need adapting.
  llvm::SmallPtrSet<StageNode*, 8> chain_ends;
  for (const auto& [edge, chain] : edge_chains) {
    chain_ends.insert(edge.first);
    chain_ends.insert(edge.second);
  }

  for (StageNode* current_stage : expanded_stages) {
    if (!chain_ends.contains(current_stage)) continue;

    StageMaterializationInfo& stage_info = plan->getStageInfo(current_stage);

    auto stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

    // The hop stage this stage exchanges data with over \p fifo_slot, or
    // nullptr if there is none.
    auto hopNeighbour = [&](mlir::Value fifo_slot,
                            bool stage_reads) -> StageNode* {
      return findFifoHop(current_stage, fifo_slot, stage_reads, endpoints,
                         edge_chains, preds);
    };

    // --- Transfer stage (kAdaptTransfer): memref/FIFO ↔ intermediate buffer
    // --- Shared helper: given a template op, its intermediate-buffer-side
    // element type and tile sizes, and its FIFO operand, check whether the
    // FIFO leads to a hop stage. If so, create the intermediate buffer spec,
    // build the edge, and register the TransferMaterializationInfo.
    // \p intermediate_is_source says whether the FIFO is the op's source.
    auto classifyTransferStage =
        [&](mlir::Operation* template_op, mlir::Type element_type,
            llvm::ArrayRef<mlir::OpFoldResult> tile_sizes,
            mlir::Value fifo_slot, bool intermediate_is_source) {
          StageNode* neighbour =
              hopNeighbour(fifo_slot, /*stage_reads=*/intermediate_is_source);
          if (!neighbour) return;
          ResourceType intermediate_resource =
              plan->getStageInfo(neighbour).stage_resource;

          llvm::SmallVector<int64_t> buffer_shape;
          for (mlir::OpFoldResult ofr : tile_sizes) {
            auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(
                ofr.dyn_cast<mlir::Attribute>());
            assert(int_attr &&
                   "Expected static tile sizes for intermediate buffer");
            buffer_shape.push_back(int_attr.getInt());
          }

          PrivateResourceSpec* buffer_spec =
              plan->resource_factory.createMemoryBuffer(
                  intermediate_resource, buffer_shape, element_type);

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
          if (edge_opt) edge = *edge_opt;

          mlir::OpBuilder builder(template_op->getContext());
          TransferMaterializationInfo* transfer_info =
              plan->transfer_factory.createFromTemplateWithBuffer(
                  template_op, edge, intermediate_resource,
                  stage_info.stage_resource, intermediate_is_source,
                  buffer_spec, builder);

          stage_info.transfers.push_back(transfer_info);
          stage_info.kind = StageMaterializationInfo::Kind::kAdaptTransfer;
          links[transfer_info] = {current_stage, neighbour, fifo_slot,
                                  /*writes_to_hop=*/!intermediate_is_source};
        };

    // Indirect transfer: dir_dst (gather) or dir_src (scatter) is the FIFO
    // slot that gets replaced by the L1 staging buffer.
    // Run this walk first so the stage is classified before the DataTransferOp
    // walk below, preventing double-classification when both op types coexist.
    stage_op.walk([&](mlir::ktdf::IndDataTransferOp ind_transfer) {
      bool is_gather = ind_transfer.isGather();
      mlir::Value fifo_side =
          is_gather ? ind_transfer.getDirDst() : ind_transfer.getDirSrc();
      auto fifo_slot_type =
          mlir::dyn_cast<mlir::ktdf::FifoSlotType>(fifo_side.getType());
      if (!fifo_slot_type) {
        ind_transfer.emitError("expected FifoSlotType on ")
            << (is_gather ? "dir_dst" : "dir_src") << " of IndDataTransferOp";
        return mlir::WalkResult::advance();
      }
      llvm::SmallVector<mlir::OpFoldResult> tile_sizes =
          is_gather ? ind_transfer.getMixedDirDstSizes()
                    : ind_transfer.getMixedDirSrcSizes();

      classifyTransferStage(ind_transfer.getOperation(),
                            fifo_slot_type.getElementType(), tile_sizes,
                            fifo_side, /*intermediate_is_source=*/!is_gather);
      return mlir::WalkResult::advance();
    });

    // Direct transfer: exactly one side is a memref (the other is a FIFO).
    // The IBR-fill DataTransferOp (both sides memref) is skipped by the guard.
    // Skip entirely if an IndDataTransferOp already classified this stage.
    if (stage_info.kind != StageMaterializationInfo::Kind::kAdaptTransfer) {
      stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
        mlir::Type src_type = transfer.getSource().getType();
        mlir::Type dest_type = transfer.getDestination().getType();

        bool src_is_memref = mlir::isa<mlir::MemRefType>(src_type);
        bool dest_is_memref = mlir::isa<mlir::MemRefType>(dest_type);

        if (src_is_memref == dest_is_memref) return mlir::WalkResult::advance();

        mlir::MemRefType memref_type =
            src_is_memref ? mlir::cast<mlir::MemRefType>(src_type)
                          : mlir::cast<mlir::MemRefType>(dest_type);
        llvm::SmallVector<mlir::OpFoldResult> tile_sizes =
            src_is_memref ? transfer.getMixedSourceSizes()
                          : transfer.getMixedDestSizes();
        mlir::Value fifo_side =
            src_is_memref ? transfer.getDestination() : transfer.getSource();

        classifyTransferStage(transfer.getOperation(),
                              memref_type.getElementType(), tile_sizes,
                              fifo_side,
                              /*intermediate_is_source=*/!src_is_memref);
        return mlir::WalkResult::advance();
      });
    }

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

      StageNode* adjacent_stage =
          hopNeighbour(fifo_slot, /*stage_reads=*/is_read);
      if (!adjacent_stage) return mlir::WalkResult::advance();

      // fifo_src = applicable_unit of the adjacent LS-unit stage (load side)
      // fifo_dest = applicable_unit of the adjacent LS-unit stage (store side)
      // For read (consuming from left): src = adjacent.applicable_unit,
      //   dest = stage.stage_resource
      // For write (producing to right): src = stage.stage_resource,
      //   dest = adjacent.applicable_unit
      const StageMaterializationInfo& adj_info =
          plan->getStageInfo(adjacent_stage);

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

      mlir::OpResult alloc_result = getFifoAllocateResult(fifo_slot);
      FifoKey fifo_key = {alloc_result ? alloc_result.getOwner() : nullptr,
                          fifo_src, fifo_dest};
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
      links[transfer_info] = {current_stage, adjacent_stage, fifo_slot,
                              /*writes_to_hop=*/!is_read};

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
// Step 2b: Retype FIFOs Between Original Stages
//===----------------------------------------------------------------------===//

/// Return the transfer registered for \p op in \p info, or nullptr.
static TransferMaterializationInfo* findTransferForOp(
    StageMaterializationInfo& info, mlir::Operation* op) {
  for (TransferMaterializationInfo* transfer : info.transfers)
    if (transfer->template_op == op) return transfer;
  return nullptr;
}

/// Return true if \p transfer replaces the FIFO operand of \p endpoint. A
/// transfer on a DataTransferOp may only replace its memref side, in which
/// case the FIFO operand is still the original one.
static bool transferRewritesFifo(const TransferMaterializationInfo* transfer,
                                 const FifoEndpoint& endpoint) {
  if (!transfer) return false;
  return endpoint.fifo_is_source ? transfer->source_private_resource != nullptr
                                 : transfer->dest_private_resource != nullptr;
}

/// Step 2b: Step 2 only rewrites FIFOs that touch a synthetic stage. A FIFO
/// whose producer and consumer are both original stages keeps the endpoints
/// from the input IR (e.g. memory-level "L1" -> "SFU"), even though Step 1 has
/// assigned an applicable unit to both stages. Retype every such FIFO to
/// (producer.applicable_unit -> consumer.applicable_unit) and register
/// transfers on both ends so the producer and consumer move to the new FIFO
/// together.
///
/// The new allocation mirrors the original fifo.allocate slot for slot, since
/// slot order is significant for FIFO semantics.
static mlir::LogicalResult retypeFifosBetweenOriginalStages(
    const FifoEndpoints& endpoints,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  const llvm::MapVector<mlir::Value, FifoEndpoint>& producers =
      endpoints.producers;
  const llvm::MapVector<mlir::Value, FifoEndpoint>& consumers =
      endpoints.consumers;
  const llvm::SmallPtrSet<mlir::Value, 4>& indirect_slots =
      endpoints.indirect_slots;

  // Classify each slot and group the ones needing a new type by the
  // fifo.allocate that defines them.
  using Endpoints = std::pair<mlir::Attribute, mlir::Attribute>;
  struct SlotToRetype {
    mlir::Value fifo_slot;
    unsigned slot_idx;
    FifoEndpoint producer;
    FifoEndpoint consumer;
    Endpoints expected;
  };
  struct AllocToRetype {
    llvm::SmallVector<SlotToRetype> slots;
    bool needs_retype = false;
  };
  llvm::MapVector<mlir::Operation*, AllocToRetype> allocs;

  for (auto& [fifo_slot, producer] : producers) {
    auto consumer_it = consumers.find(fifo_slot);
    // A slot with only one end in this pipeline has nothing to keep in sync.
    if (consumer_it == consumers.end()) continue;
    const FifoEndpoint& consumer = consumer_it->second;

    bool producer_rewritten = transferRewritesFifo(
        findTransferForOp(plan->stage_info[producer.stage], producer.op),
        producer);
    bool consumer_rewritten = transferRewritesFifo(
        findTransferForOp(plan->stage_info[consumer.stage], consumer.op),
        consumer);
    assert(producer_rewritten == consumer_rewritten &&
           "FIFO slot was rewritten on only one end; producer and consumer "
           "would use different FIFOs");
    // Both ends were already moved off this FIFO by Step 2.
    if (producer_rewritten) continue;

    std::optional<ResourceType> src_unit =
        plan->stage_info[producer.stage].applicable_unit;
    std::optional<ResourceType> dest_unit =
        plan->stage_info[consumer.stage].applicable_unit;
    if (!src_unit || !dest_unit) continue;

    auto fifo_type = mlir::cast<mlir::ktdf::FifoSlotType>(fifo_slot.getType());
    Endpoints expected = {*src_unit, *dest_unit};
    bool needs_retype =
        expected != Endpoints{fifo_type.getSrc(), fifo_type.getDest()};

    mlir::OpResult alloc_result = getFifoAllocateResult(fifo_slot);
    if (!alloc_result) {
      if (!needs_retype) continue;
      return producer.op->emitError(
          "path-expansion: cannot trace FIFO slot to its fifo.allocate");
    }
    if (needs_retype && indirect_slots.contains(fifo_slot))
      return producer.op->emitError(
          "path-expansion: retyping a FIFO used by an indirect data transfer "
          "is not supported");

    AllocToRetype& alloc = allocs[alloc_result.getOwner()];
    alloc.slots.push_back({fifo_slot, alloc_result.getResultNumber(), producer,
                           consumer, expected});
    alloc.needs_retype |= needs_retype;
  }

  for (auto& [alloc_op, alloc] : allocs) {
    if (!alloc.needs_retype) continue;

    // Every slot of the allocation must still be on the original FIFO, and
    // all of them must map to the same new endpoints; otherwise the new
    // allocation cannot mirror the original one.
    Endpoints first_expected = alloc.slots.front().expected;
    if (alloc.slots.size() != alloc_op->getNumResults() ||
        llvm::any_of(alloc.slots, [&](const SlotToRetype& slot) {
          return slot.expected != first_expected;
        }))
      return alloc_op->emitError(
          "path-expansion: cannot retype a fifo.allocate whose slots do not "
          "all connect the same pair of units");

    auto fifo_src =
        llvm::cast<ResourceType>(alloc.slots.front().expected.first);
    auto fifo_dest =
        llvm::cast<ResourceType>(alloc.slots.front().expected.second);
    llvm::SmallVector<int64_t> elements_per_slot;
    mlir::Type element_type;
    for (mlir::Type result_type : alloc_op->getResultTypes()) {
      auto fifo_type = mlir::cast<mlir::ktdf::FifoSlotType>(result_type);
      assert(!fifo_type.isDynamicNumElements() &&
             "Dynamic FIFO sizes not supported in path expansion");
      assert((!element_type || element_type == fifo_type.getElementType()) &&
             "Expected all slots of a fifo.allocate to share an element type");
      element_type = fifo_type.getElementType();
      elements_per_slot.push_back(fifo_type.getStaticNumElements());
    }
    PrivateResourceSpec* fifo_spec = plan->resource_factory.createFifo(
        fifo_src, fifo_dest, elements_per_slot, element_type);

    for (const SlotToRetype& slot : alloc.slots) {
      validateFifoSlotIndex(slot.fifo_slot, slot.slot_idx, fifo_spec);

      StageMaterializationInfo& producer_info =
          plan->stage_info[slot.producer.stage];
      StageMaterializationInfo& consumer_info =
          plan->stage_info[slot.consumer.stage];

      // Use a dummy edge when no direct edge exists between the stage
      // resources (they usually go through LS-unit nodes).
      scheduler::arch_view::RoutingGraph::NodeId src_node_id =
          arch_graph.getNodeIdForResource(producer_info.stage_resource);
      scheduler::arch_view::RoutingGraph::NodeId dst_node_id =
          arch_graph.getNodeIdForResource(consumer_info.stage_resource);
      scheduler::arch_view::RoutingGraph::EdgeInfo edge{src_node_id,
                                                        dst_node_id, 1};
      if (auto edge_opt = arch_graph.getEdgeInfo(src_node_id, dst_node_id))
        edge = *edge_opt;

      // Point the FIFO operand of one end at the new slot. A DataTransferOp
      // may already have a transfer that only replaced its memref side; reuse
      // it so the op still has a single transfer.
      auto retargetEndpoint = [&](const FifoEndpoint& endpoint,
                                  StageMaterializationInfo& info,
                                  bool is_consumer) {
        ResourceType source_resource =
            is_consumer ? fifo_src : info.stage_resource;
        ResourceType dest_resource =
            is_consumer ? info.stage_resource : fifo_dest;

        TransferMaterializationInfo* transfer =
            findTransferForOp(info, endpoint.op);
        if (!transfer) {
          transfer =
              mlir::isa<mlir::ktdf::DataTransferOp>(endpoint.op)
                  ? plan->transfer_factory.createFromTemplate(
                        endpoint.op, edge, source_resource, dest_resource)
                  : plan->transfer_factory.createFromFifoOp(
                        endpoint.op, edge, source_resource, dest_resource,
                        fifo_spec, slot.slot_idx, /*is_read=*/is_consumer);
          info.transfers.push_back(transfer);
        }
        if (is_consumer) {
          transfer->source_private_resource = fifo_spec;
          transfer->source_slot_index = slot.slot_idx;
        } else {
          transfer->dest_private_resource = fifo_spec;
          transfer->dest_slot_index = slot.slot_idx;
        }

        // The materializer only rewrites read_from_fifo / write_to_fifo in
        // kAdaptFifoKinds stages; DataTransferOps are rewritten in either
        // adapting kind.
        bool is_fifo_op = !mlir::isa<mlir::ktdf::DataTransferOp>(endpoint.op);
        if (info.kind == StageMaterializationInfo::Kind::kPreserveOriginal ||
            is_fifo_op)
          info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
      };

      retargetEndpoint(slot.producer, producer_info, /*is_consumer=*/false);
      retargetEndpoint(slot.consumer, consumer_info, /*is_consumer=*/true);

      LDBG(1) << "  Retyped FIFO slot " << slot.slot_idx << " between stage "
              << slot.producer.stage->getStageId() << " and stage "
              << slot.consumer.stage->getStageId() << " to " << fifo_src
              << " -> " << fifo_dest << "\n";
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 3: Populate Hop Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 3: Give each hop stage the transfers that move data through it.
///
/// Step 2 adapted the original stages on either side of each hop stage, so
/// one of them now writes into the hop stage's buffer or FIFO and the other
/// reads from it. Here the hop stage gets the transfer in between: from what
/// one side writes to what the other side reads. A new stage gets only these
/// transfers. An original stage that took on a hop keeps its own body and
/// gets these transfers as well.
///
/// How it works: Step 2 recorded, for each adapted op, the hop stage it
/// exchanges data with and the original FIFO slot it replaces. For each hop
/// stage, every transfer that writes into it is paired with the transfer
/// that reads the same FIFO slot from it. That pair gives the source and the
/// destination of the new transfer. Matching on the FIFO slot keeps data
/// apart when a hop stage carries data for several dependencies.
///
/// Fails if a write has no matching read or the other way around. That
/// happens when two hop stages are next to each other on one dependency,
/// which is not supported yet.
static mlir::LogicalResult populateHopStageTransfers(
    llvm::ArrayRef<StageNode*> expanded_stages, const EdgeChainMap& edge_chains,
    const TransferLinkMap& links,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  LDBG(1) << "Populating hop stage transfers for " << expanded_stages.size()
          << " stages";

  llvm::SmallPtrSet<StageNode*, 8> hops;
  for (const auto& [edge, chain] : edge_chains) hops.insert_range(chain);

  for (StageNode* hop : expanded_stages) {
    if (!hops.contains(hop)) continue;

    StageMaterializationInfo& stage_info = plan->getStageInfo(hop);
    LDBG(1) << "  Processing hop stage " << hop->getStageId()
            << " (resource: " << stage_info.stage_resource << ")\n";

    llvm::SmallVector<
        const std::pair<const TransferMaterializationInfo*, TransferLink>*>
        writes, reads;
    for (const auto& entry : links) {
      if (entry.second.hop != hop) continue;
      (entry.second.writes_to_hop ? writes : reads).push_back(&entry);
    }

    if (writes.size() != reads.size()) {
      LDBG(1) << "Hop stage " << hop->getStageId() << " has " << writes.size()
              << " transfers into it but " << reads.size()
              << " out of it; hop stages next to each other are not supported";
      return mlir::failure();
    }

    for (const auto* write : writes) {
      const TransferMaterializationInfo* prev_transfer = write->first;
      const TransferLink& write_link = write->second;

      auto read_it = llvm::find_if(reads, [&](const auto* read) {
        return read->second.fifo_slot == write_link.fifo_slot;
      });
      if (read_it == reads.end()) {
        LDBG(1) << "Hop stage " << hop->getStageId()
                << " receives data that no stage reads from it";
        return mlir::failure();
      }
      const TransferMaterializationInfo* next_transfer = (*read_it)->first;
      StageNode* reader = (*read_it)->second.stage;

      assert(prev_transfer->dest_private_resource &&
             next_transfer->source_private_resource &&
             "Transfers next to a hop stage should use its private resources");
      const PrivateResourceSpec* source_resource_spec =
          prev_transfer->dest_private_resource;
      const PrivateResourceSpec* dest_resource_spec =
          next_transfer->source_private_resource;

      ResourceType inferred_source_resource =
          plan->getStageInfo(write_link.stage).stage_resource;
      ResourceType inferred_dest_resource =
          plan->getStageInfo(reader).stage_resource;

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

      mlir::MLIRContext* ctx = write_link.stage->getOperation()->getContext();
      mlir::AffineMap source_map_for_intermediate = prev_transfer->dest_map;
      if (!source_map_for_intermediate &&
          source_resource_spec->kind ==
              PrivateResourceSpec::Kind::kMemoryBuffer) {
        source_map_for_intermediate =
            createAffineMapForIntermediateBuffer(source_resource_spec, ctx);
      }

      mlir::AffineMap dest_map_for_intermediate = next_transfer->source_map;
      if (!dest_map_for_intermediate &&
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
              source_map_for_intermediate, dest_resource_spec,
              next_transfer->source_slot_index, next_transfer->source_indices,
              next_transfer->source_sizes, dest_map_for_intermediate, ctx);

      stage_info.transfers.push_back(transfer_info);

      LDBG(1) << "    Created transfer: " << inferred_source_resource << " -> "
              << inferred_dest_resource << "\n";
    }

    if (writes.empty()) {
      LDBG(1) << "Hop stage " << hop->getStageId()
              << " has nothing to transfer; hop stages next to each other are "
                 "not supported";
      return mlir::failure();
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Orchestration
//===----------------------------------------------------------------------===//

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

static void debugPrintEdgeSegments(llvm::ArrayRef<EdgeSegment> segments) {
  for (const EdgeSegment& seg : segments) {
    llvm::dbgs() << "Path from stage " << seg.producer->getStageId()
                 << " to stage " << seg.consumer->getStageId()
                 << " (node IDs): ";
    for (auto node_id : seg.path) llvm::dbgs() << node_id << " ";
    llvm::dbgs() << "\n";
  }
}

static void debugPrintPlanningSummary(
    llvm::ArrayRef<StageNode*> expanded_stages, const PathExpansionPlan* plan) {
  llvm::dbgs() << "\n=== Planning Complete ===\n";
  llvm::dbgs() << "Total stages: " << expanded_stages.size() << "\n";
  llvm::dbgs() << "Total private resources: "
               << plan->resource_factory.getSpecs().size() << "\n";
  debugPrintResourceSpecs(llvm::dbgs(), plan->resource_factory);
}

/// Plan the expansion of a pipeline that needs it. Step 1 adds the hop stages
/// to the PipelineTree and rewires the dependencies. Steps 2, 2b and 3 record
/// in \p plan how each stage must be rewritten: the buffers and FIFOs to
/// create, and the transfers to adapt or add. Nothing is changed in the IR
/// here; the materializer does that afterwards from the tree and the plan.
static mlir::LogicalResult buildExpansionPlan(
    PipelineTree& tree, PipelineNode* pipeline,
    llvm::ArrayRef<EdgeSegment> segments,
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  // STEP 1: Add hop stages along each dependency
  LDBG(1) << "\n=== Step 1: Add Stages Along Each Path ===\n";
  EdgeChainMap edge_chains;
  if (mlir::failed(expandStageEdges(tree, pipeline, segments, sorted_stages,
                                    arch_graph, plan, next_stage_id,
                                    edge_chains))) {
    return mlir::failure();
  }
  if (mlir::failed(orderStagesByDependencies(pipeline))) {
    LDBG(1) << "Giving hops to existing stages created a dependency cycle";
    return mlir::failure();
  }
  llvm::FailureOr<llvm::SmallVector<StageNode*>> expanded_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(expanded_stages_or)) {
    return mlir::failure();
  }
  llvm::SmallVector<StageNode*> expanded_stages = *expanded_stages_or;

  // Each unit must still run at most one stage.
  llvm::DenseSet<ResourceType> used_units;
  for (StageNode* stage : expanded_stages) {
    std::optional<ResourceType> unit =
        plan->getStageInfo(stage).applicable_unit;
    if (unit && !used_units.insert(*unit).second) {
      LDBG(1) << "Unit " << *unit << " runs more than one stage";
      return mlir::failure();
    }
  }
  StagePredecessorMap preds = buildPredecessorMap(expanded_stages);
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 1");
  });

  FifoEndpoints fifo_endpoints;
  if (mlir::failed(collectFifoEndpoints(sorted_stages, fifo_endpoints))) {
    return mlir::failure();
  }

  // STEP 2: Classify original stages and create private resources
  LDBG(1) << "\n=== Step 2: Classify Original Stages ===\n";
  TransferLinkMap links;
  if (mlir::failed(classifyOriginalStages(expanded_stages, preds,
                                          fifo_endpoints, edge_chains,
                                          arch_graph, plan, links))) {
    return mlir::failure();
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 2");
  });

  // STEP 2b: Retype FIFOs whose producer and consumer are both original stages
  LDBG(1) << "\n=== Step 2b: Retype FIFOs Between Original Stages ===\n";
  if (mlir::failed(
          retypeFifosBetweenOriginalStages(fifo_endpoints, arch_graph, plan))) {
    return mlir::failure();
  }
  LDBG_OS(1, [&](llvm::raw_ostream& os) {
    debugPrintStageList(os, expanded_stages, plan, "After Step 2b");
  });

  // STEP 3: Populate hop stage transfers
  LDBG(1) << "\n=== Step 3: Populate Hop Stage Transfers ===\n";
  if (mlir::failed(populateHopStageTransfers(expanded_stages, edge_chains,
                                             links, arch_graph, plan))) {
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

  // PREP 1: Sort and validate pipeline stages, and find each stage's side
  // (load, compute or store).
  StageSideMap sides;
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      sortAndValidateStages(tree, pipeline, arch_graph, sides);
  if (mlir::failed(sorted_stages_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  LLVM_DEBUG(debugPrintInitialPlannerState(pipeline, sorted_stages));

  // PREP 2: Assign resources
  assignOriginalStageResources(sorted_stages, sides, plan.get());

  // PREP 3: Find the routing path for every dependency between stages
  llvm::FailureOr<llvm::SmallVector<EdgeSegment>> segments_or =
      buildEdgeSegments(sorted_stages, sides, arch_graph, plan.get());
  if (mlir::failed(segments_or)) {
    LDBG(1) << "No resource path found\n";
    return nullptr;
  }
  llvm::SmallVector<EdgeSegment> segments = std::move(*segments_or);

  LLVM_DEBUG(debugPrintEdgeSegments(segments));

  // PREP 4: Assign units to the original stages; each must get its own
  if (mlir::failed(assignOriginalStageUnits(segments, sorted_stages, arch_graph,
                                            plan.get()))) {
    return nullptr;
  }

  // PREP 5: Check whether path expansion is needed
  if (!needsExpansion(segments, arch_graph)) {
    plan->changed = false;
    LDBG(1) << "Pipeline already legal";
    return plan;
  }

  plan->changed = true;

  int next_stage_id = static_cast<int>(sorted_stages.size());
  if (mlir::failed(buildExpansionPlan(tree, pipeline, segments, sorted_stages,
                                      arch_graph, plan.get(), next_stage_id))) {
    return nullptr;
  }

  return plan;
}

}  // namespace scheduler

// Made with Bob
