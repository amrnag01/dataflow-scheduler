//===-- FoldFifoRelayStages.cpp ---------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep

#define PASS_NAME "fold-fifo-relay-stages"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Fold FIFO Relay Stages pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_FOLDFIFORELAYSTAGESPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

/// Discardable attributes that block the fold of a pair when both transfers
/// carry them with different values.
constexpr llvm::StringLiteral kNonMergeableAttrs[] = {
    "dataflow_scheduler.throttle",
};

/// A transfer into a FIFO slot and the transfer out of it.
struct RelayPair {
  mlir::ktdf::DataTransferOp in;
  mlir::ktdf::DataTransferOp out;
};

/// Returns the transfers in @p stage , or std::nullopt if it holds anything
/// else.
[[nodiscard]] auto getTransfers(mlir::ktdf::StageOp stage)
    -> std::optional<llvm::SmallVector<mlir::ktdf::DataTransferOp>> {
  llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
  for (auto& op : *stage.getBody()) {
    auto transfer = llvm::dyn_cast<mlir::ktdf::DataTransferOp>(op);
    if (!transfer) {
      return std::nullopt;
    }
    transfers.push_back(transfer);
  }
  if (transfers.empty()) {
    return std::nullopt;
  }
  return transfers;
}

[[nodiscard]] auto getMemorySpace(mlir::Value memref) -> mlir::Attribute {
  return llvm::cast<mlir::MemRefType>(memref.getType()).getMemorySpace();
}

/// Merges the discardable attributes of @p in and @p out , preferring those of
/// @p in . Returns std::nullopt if a non-mergeable attribute conflicts.
[[nodiscard]] auto mergeAttrs(mlir::ktdf::DataTransferOp in,
                              mlir::ktdf::DataTransferOp out)
    -> std::optional<mlir::NamedAttrList> {
  for (const auto name : kNonMergeableAttrs) {
    const auto in_attr = in->getDiscardableAttr(name);
    const auto out_attr = out->getDiscardableAttr(name);
    if (in_attr && out_attr && in_attr != out_attr) {
      return std::nullopt;
    }
  }

  mlir::NamedAttrList merged(in->getDiscardableAttrDictionary());
  for (const auto attr : out->getDiscardableAttrs()) {
    if (!merged.get(attr.getName())) {
      merged.append(attr);
    }
  }
  return merged;
}

/// Merges the applicable units of @p in and @p out . Returns std::nullopt if
/// both are set and do not intersect.
[[nodiscard]] auto mergeUnits(mlir::ktdf::StageOp in, mlir::ktdf::StageOp out)
    -> std::optional<mlir::ArrayAttr> {
  const auto in_units = in.getApplicableUnitsAttr();
  const auto out_units = out.getApplicableUnitsAttr();
  if (!in_units || !out_units) {
    return in_units ? in_units : out_units;
  }

  llvm::SmallVector<mlir::Attribute> common;
  for (const auto unit : in_units) {
    if (llvm::is_contained(out_units.getValue(), unit)) {
      common.push_back(unit);
    }
  }
  if (common.empty()) {
    return std::nullopt;
  }
  return mlir::ArrayAttr::get(in.getContext(), common);
}

/// Folds the stage pair that relays data from memory to memory through FIFOs,
/// starting at @p in , into the consuming stage.
///
/// @return Whether the fold happened.
auto foldRelay(mlir::RewriterBase& rewriter, mlir::ktdf::StageOp in) -> bool {
  const auto in_transfers = getTransfers(in);
  if (!in_transfers) {
    return false;
  }
  const auto reject = [&](llvm::StringRef reason) -> bool {
    LDBG() << "not folding " << mlir::OpWithFlags(in, kSkipRegions) << ": "
           << reason;
    return false;
  };

  // Pair every transfer into a FIFO slot with the single transfer out of it,
  // all in one consuming stage.
  mlir::ktdf::StageOp out;
  llvm::SmallVector<RelayPair> pairs;
  for (auto transfer : *in_transfers) {
    if (!transfer.isSourceMemRef() || !transfer.isDestFifo()) {
      return false;
    }

    auto slot = transfer.getDestination();
    if (!slot.hasNUses(2)) {
      return reject("FIFO slot has other users");
    }
    mlir::ktdf::DataTransferOp reader;
    for (auto* user : slot.getUsers()) {
      if (user != transfer) {
        reader = llvm::dyn_cast<mlir::ktdf::DataTransferOp>(user);
      }
    }
    if (!reader || reader.getSource() != slot || !reader.isDestMemRef()) {
      return reject("FIFO slot is not read into memory");
    }

    auto stage = llvm::dyn_cast<mlir::ktdf::StageOp>(reader->getParentOp());
    if (!stage || stage == in || (out && stage != out)) {
      return reject("FIFO slots are not read by a single other stage");
    }
    out = stage;
    pairs.push_back({transfer, reader});
  }

  // The consuming stage must hold nothing but the other half of the pairs.
  const auto out_transfers = getTransfers(out);
  if (!out_transfers || out_transfers->size() != pairs.size()) {
    return reject("consumer holds more than the relayed transfers");
  }

  // All FIFOs connect the same pair of resources, and all transfers move
  // between the same pair of distinct memories. Distinct memories ensure no
  // destination aliases a source, so the pairs can execute in any order.
  const auto first_slot = llvm::cast<mlir::ktdf::FifoSlotType>(
      pairs.front().in.getDestination().getType());
  const auto src_space = getMemorySpace(pairs.front().in.getSource());
  const auto dest_space = getMemorySpace(pairs.front().out.getDestination());
  if (src_space == dest_space) {
    return reject("source and destination memories are the same");
  }
  for (auto [transfer_in, transfer_out] : pairs) {
    const auto slot = llvm::cast<mlir::ktdf::FifoSlotType>(
        transfer_in.getDestination().getType());
    if (slot.getSrc() != first_slot.getSrc() ||
        slot.getDest() != first_slot.getDest() ||
        getMemorySpace(transfer_in.getSource()) != src_space ||
        getMemorySpace(transfer_out.getDestination()) != dest_space) {
      return reject("transfers do not share their end points");
    }

    // What goes into the FIFO must be what comes out of it.
    const auto pushed = transfer_in.getMixedDestSizes();
    const auto popped = transfer_out.getMixedSourceSizes();
    if (pushed.size() != popped.size() ||
        !llvm::all_of(llvm::zip_equal(pushed, popped), [](auto sizes) {
          return mlir::isEqualConstantIntOrValue(std::get<0>(sizes),
                                                 std::get<1>(sizes));
        })) {
      return reject("FIFO slot sizes do not match");
    }
  }

  // The tokens the producer signals must only be awaited by the consumer.
  for (auto token : in.getDependsOut()) {
    for (auto& use : token.getUses()) {
      if (use.getOwner() == in) {
        continue;
      }
      if (use.getOwner() != out || !out.isInDependency(use)) {
        return reject("producer token is awaited by another stage");
      }
    }
  }

  const auto units = mergeUnits(in, out);
  if (!units) {
    return reject("applicable units do not intersect");
  }
  llvm::SmallVector<mlir::NamedAttrList> attrs;
  for (auto [transfer_in, transfer_out] : pairs) {
    auto merged = mergeAttrs(transfer_in, transfer_out);
    if (!merged) {
      return reject("transfers carry conflicting attributes");
    }
    attrs.push_back(std::move(*merged));
  }

  LDBG() << "folding " << pairs.size() << " relay(s) of "
         << mlir::OpWithFlags(in, kSkipRegions) << " into "
         << mlir::OpWithFlags(out, kSkipRegions);

  // Rewrite the consumer into memory-to-memory transfers. Inserting at the
  // consumer keeps the stages in dependency order, and all operands are
  // defined outside of both stages.
  for (auto [pair, merged] : llvm::zip_equal(pairs, attrs)) {
    rewriter.setInsertionPoint(pair.out);
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, rewriter.getFusedLoc({pair.in.getLoc(), pair.out.getLoc()}),
        pair.in.getSource(), pair.in.getSourceMap().value_or(mlir::AffineMap{}),
        pair.in.getSourceIndices(), pair.in.getMixedSourceSizes(),
        pair.out.getDestination(),
        pair.out.getDestMap().value_or(mlir::AffineMap{}),
        pair.out.getDestIndices(), pair.out.getMixedDestSizes());
    transfer->setDiscardableAttrs(merged.getDictionary(rewriter.getContext()));
    rewriter.eraseOp(pair.out);
  }

  llvm::SetVector<mlir::Value> depends_in(in.getDependsIn().begin(),
                                          in.getDependsIn().end());
  for (auto token : out.getDependsIn()) {
    if (!llvm::is_contained(in.getDependsOut(), token)) {
      depends_in.insert(token);
    }
  }
  rewriter.modifyOpInPlace(out, [&]() {
    out.setDependsIn(depends_in.getArrayRef());
    if (*units) {
      out.setApplicableUnitsAttr(*units);
    } else {
      out.removeApplicableUnitsAttr();
    }
  });
  rewriter.eraseOp(in);
  return true;
}

/// Drops the slots without uses from @p alloc , unless it has no slots in use
/// at all, in which case it is left for erasure.
void shrinkFifoAllocation(mlir::RewriterBase& rewriter,
                          mlir::ktdf::FifoAllocateOp alloc) {
  if (alloc->use_empty() ||
      llvm::none_of(alloc.getSlots(),
                    [](mlir::Value slot) { return slot.use_empty(); })) {
    return;
  }

  llvm::SmallVector<mlir::Value> kept_slots;
  llvm::SmallVector<mlir::Type> kept_types;
  llvm::SmallVector<mlir::Value> kept_sizes;
  auto dynamic_sizes = alloc.getDynamicSizes().begin();
  for (auto slot : alloc.getSlots()) {
    const auto type = llvm::cast<mlir::ktdf::FifoSlotType>(slot.getType());
    const auto size =
        type.isDynamicNumElements() ? *dynamic_sizes++ : mlir::Value{};
    if (slot.use_empty()) {
      continue;
    }
    kept_slots.push_back(slot);
    kept_types.push_back(type);
    if (size) {
      kept_sizes.push_back(size);
    }
  }

  rewriter.setInsertionPoint(alloc);
  auto shrunk = mlir::ktdf::FifoAllocateOp::create(rewriter, alloc.getLoc(),
                                                   kept_types, kept_sizes);
  shrunk->setDiscardableAttrs(alloc->getDiscardableAttrDictionary());
  rewriter.replaceAllUsesWith(kept_slots, shrunk.getSlots());
  rewriter.eraseOp(alloc);
}

struct FoldFifoRelayStagesPass
    : impl::FoldFifoRelayStagesPassBase<FoldFifoRelayStagesPass> {
  using FoldFifoRelayStagesPassBase::FoldFifoRelayStagesPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    mlir::IRRewriter rewriter(&getContext());
    getOperation()->walk([&](mlir::ktdf::PipelineOp pipeline) {
      // A producer only ever transfers into FIFOs and a consumer only out of
      // them, so no folded stage can take part in another fold.
      auto changed = false;
      for (auto stage : llvm::to_vector(pipeline.getStages())) {
        if (foldRelay(rewriter, stage)) {
          ++num_folded;
          changed = true;
        }
      }
      if (!changed) {
        return;
      }

      // Drop the private slots and tokens left without users.
      mlir::ktdf::PrivateBuilder::canonicalize(rewriter, pipeline);
      if (auto private_op = pipeline.getPrivateOp(); private_op) {
        for (auto alloc : llvm::to_vector(
                 private_op.getBody()->getOps<mlir::ktdf::FifoAllocateOp>())) {
          shrinkFifoAllocation(rewriter, alloc);
        }
      }
    });
  }
};

}  // namespace
