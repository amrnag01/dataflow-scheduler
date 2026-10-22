//===-- MaterializeRegisters.cpp -------------------------*- c++ -*-===//
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
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/RegionUtils.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Utils/Utils.h"

namespace scheduler {
#define GEN_PASS_DEF_MATERIALIZEREGISTERSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace mlir;

namespace scheduler {
namespace {

/// Gets the mapped constants the body of \p generic reads, in reading order.
///
/// Found by use rather than by where they stand: folding materializes a
/// constant in the entry block of its region, so by now most are outside the
/// body a pattern put them in. The mapping marks one as belonging in a register
/// rather than being an operand of the arithmetic.
auto getMappedConstants(linalg::GenericOp generic)
    -> SmallVector<arith::ConstantOp> {
  // Ordered, because the order they are found in is the operand order that
  // names each of them a register.
  SetVector<Operation*> seen;
  generic.getBody()->walk([&](Operation* op) {
    for (const auto operand : op->getOperands()) {
      const auto constant = operand.getDefiningOp<arith::ConstantOp>();
      if (!constant) continue;
      if (!ktdf_arch::getProperty<ktdf_arch::MapsToAttr>(constant)) continue;
      seen.insert(constant);
    }
  });

  SmallVector<arith::ConstantOp> result;
  for (auto* op : seen) result.push_back(cast<arith::ConstantOp>(op));
  return result;
}

/// Turns \p constant into a register in front of \p generic.
///
/// A template reads a whole vector out of a register, so the register is filled
/// with the value repeated across its lanes, as many as \p compute has. Returns
/// it, for the body to read instead of the constant.
auto materializeRegister(arith::ConstantOp constant, linalg::GenericOp generic,
                         ktdf_arch::ExecutionUnitOp compute,
                         RewriterBase& rewriter) -> Value {
  const auto maps_to = ktdf_arch::getProperty<ktdf_arch::MapsToAttr>(constant);
  const auto element = constant.getType();
  const auto lanes = getVectorLanes(element, compute);

  const auto register_type =
      MemRefType::get({lanes}, element, MemRefLayoutAttrInterface{}, maps_to);

  rewriter.setInsertionPoint(generic);
  auto reg =
      memref::AllocOp::create(rewriter, constant.getLoc(), register_type);
  linalg::FillOp::create(rewriter, constant.getLoc(),
                         ValueRange{constant.getResult()}, ValueRange{reg});
  return reg;
}

/// Gets the registers the body of \p generic allocates for itself, in order.
///
/// A pattern allocates the scratch a template computes in where it matched, so
/// that lands in the body too. Only allocations sized entirely by their type
/// are taken; one the body computes a size for could not move out of it.
auto getBodyAllocations(linalg::GenericOp generic) -> SmallVector<Operation*> {
  SmallVector<Operation*> result;
  generic.getBody()->walk([&](Operation* op) {
    if (!llvm::isa<memref::AllocaOp, memref::AllocOp>(op)) return;
    if (op->getNumOperands() != 0) return;
    result.push_back(op);
  });
  return result;
}

/// Gets how many elements of its tile \p generic covers, zero if it is not
/// static.
///
/// The parallel dimensions only. A register holds one element per lane of what
/// the compute works on at once, and a reduction dimension is iterated rather
/// than held -- counting it would size the register to every step of the
/// reduction instead of the accumulator it carries.
auto getTileSize(linalg::GenericOp generic) -> int64_t {
  const auto iterators = generic.getIteratorTypesArray();

  // Only the parallel dimensions inside the last reduction are lanes of the
  // register. One outside it is a loop around the whole reduction, and every
  // turn of it writes the register's same position.
  const auto ranges = generic.getStaticLoopRanges();
  int64_t result = 1;
  for (int64_t dim = static_cast<int64_t>(ranges.size()) - 1; dim >= 0; --dim) {
    if (iterators[dim] == utils::IteratorType::reduction) break;
    if (ShapedType::isDynamic(ranges[dim])) return 0;
    if (llvm::MulOverflow<int64_t>(result, ranges[dim], result)) return 0;
  }
  return result;
}

/// Sizes \p alloc to the tile \p generic covers and moves it in front of it.
///
/// A pattern says the element a register holds and leaves the count to here,
/// where the tile says it: what the template computes on is the tile, whether
/// that is one register or several. An allocation that already carries a shape
/// is taken at its word and only moved.
///
/// \p zero is the index of the lane the body's own accesses land on, made here
/// on the first allocation that needs one.
auto hoistAllocation(Operation* alloc, linalg::GenericOp generic,
                     ktdf_arch::ExecutionUnitOp compute, Value& zero,
                     RewriterBase& rewriter) -> LogicalResult {
  const auto type = cast<MemRefType>(alloc->getResult(0).getType());
  if (type.getRank() != 0) {
    rewriter.moveOpBefore(alloc, generic);
    return success();
  }

  for (auto* const user : alloc->getUsers()) {
    if (!llvm::isa<memref::StoreOp, memref::LoadOp, ktdf::OpaqueOp>(user)) {
      return alloc->emitError("unable to hoist allocation")
                 .attachNote(user->getLoc())
             << "user can't be vectorized";
    }
  }

  const auto element = type.getElementType();
  const auto tile = getTileSize(generic);
  if (tile == 0) {
    return generic.emitError("the tile is not a static number of elements");
  }

  // A register holds whole lanes of the element, so a tile of more than one has
  // to be a whole number of them: otherwise the template would read a lane it
  // never wrote. Less than one register still gets one -- what the body works
  // on is a position in it.
  const auto lanes = getVectorLanes(element, compute);
  int64_t held = tile;
  if (tile < lanes) {
    held = lanes;
  } else if (tile % lanes != 0) {
    return alloc->emitError("a tile of ")
           << tile << " does not divide into registers of " << lanes << " "
           << element;
  }

  const auto register_type = MemRefType::get(
      {held}, element, MemRefLayoutAttrInterface{}, type.getMemorySpace());

  rewriter.setInsertionPoint(generic);
  // The kind the body asked for is kept: what hoists a register's fill out of a
  // loop hoists an alloc and leaves an alloca alone, so the pattern says which
  // it wants by which one it wrote.
  Value reg =
      llvm::isa<memref::AllocOp>(alloc)
          ? memref::AllocOp::create(rewriter, alloc->getLoc(), register_type)
                .getResult()
          : memref::AllocaOp::create(rewriter, alloc->getLoc(), register_type)
                .getResult();
  if (!zero) {
    zero = arith::ConstantIndexOp::create(rewriter, alloc->getLoc(), 0);
  }

  // What the body writes and reads is one element, and it becomes lane zero of
  // the register. The lowering below turns either into the whole of it.
  for (auto* user : llvm::to_vector(alloc->getUsers())) {
    rewriter.setInsertionPoint(user);
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      // A value from outside the body is the same for every element, so it
      // belongs to the whole register rather than to lane zero. Runtime scalars
      // such as the base and the stride of an address arrive this way, hoisted
      // out of the body before this pass runs.
      const auto stored = store.getValueToStore();
      if (areValuesDefinedAbove(ValueRange{stored}, generic.getBodyRegion())) {
        rewriter.setInsertionPoint(generic);
        linalg::FillOp::create(rewriter, store.getLoc(), ValueRange{stored},
                               ValueRange{reg});
      } else {
        memref::StoreOp::create(rewriter, store.getLoc(), stored, reg,
                                ValueRange{zero});
      }
      rewriter.eraseOp(store);
      continue;
    }
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      auto value = memref::LoadOp::create(rewriter, load.getLoc(), reg,
                                          ValueRange{zero});
      rewriter.replaceOp(load, value.getResult());
    }
  }

  // Whoever is left takes the register as it stands -- the opaque, which reads
  // and writes the whole of it.
  rewriter.replaceOp(alloc, reg);
  return success();
}

/// Whether \p reader takes a float rather than the register: arithmetic does,
/// a template call does not.
auto wantsTheValue(Operation* reader) -> bool {
  Dialect* const dialect = reader->getDialect();
  return isa_and_nonnull<arith::ArithDialect>(dialect) ||
         isa_and_nonnull<math::MathDialect>(dialect);
}

/// Reads the first lane of \p reg in front of \p reader. The fill put the same
/// value in every lane, so any lane would do.
auto loadFirstLane(Value reg, Operation* reader, RewriterBase& rewriter)
    -> Value {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(reader);
  Value zero = arith::ConstantIndexOp::create(rewriter, reader->getLoc(), 0);
  return memref::LoadOp::create(rewriter, reader->getLoc(), reg,
                                ValueRange{zero});
}

/// Makes the registers \p generic uses real in front of its body.
auto materializeRegisters(linalg::GenericOp generic,
                          ktdf_arch::Mapping& mapping, RewriterBase& rewriter)
    -> LogicalResult {
  const auto constants = getMappedConstants(generic);
  const auto allocations = getBodyAllocations(generic);
  if (constants.empty() && allocations.empty()) {
    // Nothing to size, so nothing to say about where this one computes.
    return success();
  }

  // We assume the register is for the compute that the generic is mapped to,
  // which is defaulted if it is not yet mapped.
  auto compute = mapping.getOrMap<ktdf_arch::ExecutionUnitOp>(
      generic, mapping.byKind().getDefaultCompute().getKind());
  if (!compute) {
    return generic.emitError("no compute resource is mapped to size registers");
  }

  for (auto constant : constants) {
    // Only one still inside has to move. Moving one that is already out could
    // put it after something else that reads it.
    if (generic->isProperAncestor(constant)) {
      rewriter.moveOpBefore(constant, generic);
    }
    const auto reg = materializeRegister(constant, generic, compute, rewriter);
    if (!reg) return failure();

    // Whatever read the constant reads the register now, where it stands: the
    // body is not isolated from above. Arithmetic reads a lane out of it.
    SmallVector<OpOperand*> takes_the_value;
    for (OpOperand& use : constant.getResult().getUses()) {
      if (generic->isProperAncestor(use.getOwner()) &&
          wantsTheValue(use.getOwner())) {
        takes_the_value.push_back(&use);
      }
    }
    rewriter.replaceUsesWithIf(constant.getResult(), reg, [&](OpOperand& use) {
      return generic->isProperAncestor(use.getOwner()) &&
             !wantsTheValue(use.getOwner());
    });
    for (OpOperand* use : takes_the_value) {
      use->set(loadFirstLane(reg, use->getOwner(), rewriter));
    }
  }

  // The scratch moves out sized to the tile, and the body goes on reading it
  // there. Handing it in as an operand is not open: a generic takes either
  // tensors or buffers throughout, and the data here is tensors.
  Value zero;
  for (auto* alloc : allocations) {
    if (failed(hoistAllocation(alloc, generic, compute, zero, rewriter))) {
      return failure();
    }
  }

  return success();
}

struct MaterializeRegistersPass
    : public impl::MaterializeRegistersPassBase<MaterializeRegistersPass> {
  using MaterializeRegistersPassBase::MaterializeRegistersPassBase;

  void runOnOperation() override {
    // Obtain the default device, emitting a diagnostic on failure.
    const auto& default_device = getAnalysis<ktdf_arch::DefaultDevice>();
    if (!default_device) {
      signalPassFailure();
      return;
    }
    ktdf_arch::Mapping mapping(default_device.getRef());
    if (!mapping.byKind().getDefaultCompute()) {
      mapping.getDevice().getDeclaration().emitError(
          "no (unambiguous) default compute resource");
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    const auto result = getOperation()->walk([&](linalg::GenericOp generic) {
      if (failed(materializeRegisters(generic, mapping, rewriter))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace scheduler
