//===-- LoopIVAffineMulStrengthReduction.cpp --------------------*- c++ -*-===//
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
// Two complementary strength reductions are performed in a single pass:
//
// 1. affine.apply weighted-sum reduction
//    Handles any affine.apply whose single result is a weighted sum of IVs of
//    enclosing scf.for loops:
//      (d0, d1, ..., dn) -> (d0*N0 + d1*N1 + ... + dn*Nn)
//    The transformation threads one iter-arg per participating loop, where each
//    iter-arg carries the partial sum through that loop.  Non-participating
//    loops between consecutive participating loops pass the value unchanged.
//    The innermost iter-arg replaces the affine.apply.
//
// 2. arith.muli IV x loop-invariant-factor reduction
//    Handles bare arith.muli whose one operand is the IV of an enclosing
//    scf.for and the other is any value defined outside that loop:
//      %m = arith.muli %iv, %factor   (or %factor, %iv)
//    An iter-arg is added to the loop, initialized to (lb * factor) and
//    incremented by (step * factor) each iteration.  The arith.muli is
//    replaced by the iter-arg.  Multiple multiplications of the same IV by the
//    same factor within one loop are coalesced into a single iter-arg.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "loop-iv-affine-mul-strength-reduction"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable affine-apply IV mul strength reduction"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_LOOPIVAFFINEMULSTRENGTHREDUCTIONPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

// ---------------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------------

/// Try to decompose an AffineExpr as a sum of terms `dim(i) * Ni` where each
/// Ni is an integer constant.  Returns a map from dimension index to
/// coefficient on success, or std::nullopt if the expression contains any term
/// that is not of that form (e.g. a symbol, a non-dim sub-expression, or a
/// non-constant coefficient).
static std::optional<llvm::SmallDenseMap<unsigned, int64_t>>
matchWeightedDimSum(AffineExpr expr) {
  llvm::SmallDenseMap<unsigned, int64_t> coeffs;

  std::function<bool(AffineExpr)> visit = [&](AffineExpr e) -> bool {
    if (auto add = dyn_cast<AffineBinaryOpExpr>(e)) {
      if (add.getKind() == AffineExprKind::Add)
        return visit(add.getLHS()) && visit(add.getRHS());
    }
    // Term must be either: dim(i)  or  dim(i) * cst  or  cst * dim(i).
    if (auto dim = dyn_cast<AffineDimExpr>(e)) {
      coeffs[dim.getPosition()] += 1;
      return true;
    }
    if (auto mul = dyn_cast<AffineBinaryOpExpr>(e)) {
      if (mul.getKind() != AffineExprKind::Mul) return false;
      auto dim_lhs = dyn_cast<AffineDimExpr>(mul.getLHS());
      auto cst_rhs = dyn_cast<AffineConstantExpr>(mul.getRHS());
      if (dim_lhs && cst_rhs) {
        coeffs[dim_lhs.getPosition()] += cst_rhs.getValue();
        return true;
      }
      // cst * dim form
      auto cst_lhs = dyn_cast<AffineConstantExpr>(mul.getLHS());
      auto dim_rhs = dyn_cast<AffineDimExpr>(mul.getRHS());
      if (cst_lhs && dim_rhs) {
        coeffs[dim_rhs.getPosition()] += cst_lhs.getValue();
        return true;
      }
    }
    return false;
  };

  if (!visit(expr)) return std::nullopt;
  return coeffs;
}

/// Walk up from `op` and return all enclosing scf.for loops, innermost first.
static SmallVector<scf::ForOp> enclosingLoops(Operation* op) {
  SmallVector<scf::ForOp> loops;
  for (Operation* p = op->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p)) loops.push_back(f);
  return loops;
}

// ---------------------------------------------------------------------------
// Per-candidate transformation
// ---------------------------------------------------------------------------

/// Describes one participating loop: the scf.for loop, the constant
/// coefficient for its IV in the affine map, and its position in the
/// innermost-to-outermost enclosing-loop list (used to count middle loops).
///
/// lb and step are snapshotted at collection time.  replaceWithAdditionalYields
/// invalidates the scf::ForOp handle it is called on, but moves all inner ops
/// intact into the replacement loop — so inner loop handles remain valid.
/// Outer loop handles of the *same* candidate become stale after their own
/// replaceWithAdditionalYields call, hence the snapshots.
struct ParticipatingLoop {
  scf::ForOp loop;  // live until this loop's own replaceWithAdditionalYields
  Value lb;         // snapshotted lower bound
  Value step;       // snapshotted step
  int64_t coeff;    // Ni in `iv * Ni`
  int enclosing_index;  // index in the innermost-to-outermost enclosing list
};

/// Add one pass-through iter-arg to `loop`.  Returns the new BlockArgument.
static FailureOr<BlockArgument> addPassthroughIterArg(IRRewriter& rewriter,
                                                      scf::ForOp loop,
                                                      Value init) {
  auto yield_fn = [](OpBuilder& b, Location loc,
                     ArrayRef<BlockArgument> args) -> SmallVector<Value> {
    return {args[0]};
  };
  FailureOr<LoopLikeOpInterface> res = loop.replaceWithAdditionalYields(
      rewriter, {init}, /*replaceInitOperandUsesInLoop=*/false, yield_fn);
  if (failed(res)) return failure();
  return cast<scf::ForOp>(*res).getRegionIterArgs().back();
}

/// Add an incrementing iter-arg to `loop`.  Returns the new BlockArgument.
static FailureOr<BlockArgument> addIncrementalIterArg(IRRewriter& rewriter,
                                                      scf::ForOp loop,
                                                      Value init,
                                                      Value increment) {
  auto yield_fn = [&](OpBuilder& b, Location loc,
                      ArrayRef<BlockArgument> args) -> SmallVector<Value> {
    return {arith::AddIOp::create(b, loc, args[0], increment)};
  };
  FailureOr<LoopLikeOpInterface> res = loop.replaceWithAdditionalYields(
      rewriter, {init}, /*replaceInitOperandUsesInLoop=*/false, yield_fn);
  if (failed(res)) return failure();
  return cast<scf::ForOp>(*res).getRegionIterArgs().back();
}

/// Transform one affine.apply candidate.  `participating` is ordered
/// outermost-to-innermost.  Returns true if the IR was changed.
///
/// Handle-invalidation strategy
/// ----------------------------
/// replaceWithAdditionalYields(loop) invalidates `loop` but moves all inner
/// ops (including inner loops) intact into the replacement.  We therefore
/// process loops outermost-to-innermost: when we call replaceWithAdditional
/// Yields on loop[k], all loop[k+1..n] handles are still valid because they
/// live inside loop[k]'s body which was moved.
///
/// Middle (non-participating) loops between loop[k-1] and loop[k] are found
/// by walking the parent chain of loop[k] at the time we process it (before
/// we call replaceWithAdditionalYields on loop[k] itself).  Because loop[k]
/// is alive at this point, its parent chain contains the live replacements of
/// any outer loops that have already been processed.
///
/// lb and step are snapshotted in ParticipatingLoop so we do not call
/// getLowerBound()/getStep() after the handle becomes invalid.
static bool transformCandidate(affine::AffineApplyOp apply,
                               ArrayRef<ParticipatingLoop> participating,
                               IRRewriter& rewriter) {
  Location loc = apply.getLoc();

  // Insert %c0 before the outermost participating loop.  At this point all
  // loop handles in `participating` are still alive.
  rewriter.setInsertionPoint(participating.front().loop);
  Value current_val = arith::ConstantIndexOp::create(rewriter, loc, 0);

  Value innermost_arg;

  for (size_t k = 0; k < participating.size(); ++k) {
    const ParticipatingLoop& pl = participating[k];

    // Find middle loops between participating[k-1] and participating[k].
    // Walk the live parent chain of pl.loop (which is alive at this point).
    // The number of middle loops equals the gap in enclosing_index minus 1.
    int num_middle =
        (k == 0)
            ? 0
            : (participating[k - 1].enclosing_index - pl.enclosing_index - 1);

    SmallVector<scf::ForOp> middle;
    if (num_middle > 0) {
      Operation* p = pl.loop->getParentOp();
      for (int m = 0; m < num_middle && p; p = p->getParentOp())
        if (auto f = dyn_cast<scf::ForOp>(p)) {
          middle.push_back(f);
          ++m;
        }
    }
    // Thread current_val through middle loops outermost-to-innermost.
    for (scf::ForOp mid : llvm::reverse(middle)) {
      FailureOr<BlockArgument> arg =
          addPassthroughIterArg(rewriter, mid, current_val);
      if (failed(arg)) return false;
      current_val = *arg;
    }

    // Create init/increment arithmetic before calling replaceWithAdditional
    // Yields (which will invalidate pl.loop).  Use snapshotted lb/step.
    rewriter.setInsertionPoint(pl.loop);
    Value cst_n = arith::ConstantIndexOp::create(rewriter, loc, pl.coeff);
    Value own_lb = rewriter.createOrFold<arith::MulIOp>(loc, pl.lb, cst_n);
    Value init = rewriter.createOrFold<arith::AddIOp>(loc, current_val, own_lb);
    Value increment = rewriter.createOrFold<arith::MulIOp>(loc, pl.step, cst_n);

    FailureOr<BlockArgument> arg =
        addIncrementalIterArg(rewriter, pl.loop, init, increment);
    if (failed(arg)) return false;
    current_val = *arg;
    innermost_arg = *arg;
  }

  rewriter.replaceOp(apply, innermost_arg);
  return true;
}

// ---------------------------------------------------------------------------
// arith.muli IV x loop-invariant-factor transformation
// ---------------------------------------------------------------------------

/// For a single scf.for loop, find all `arith.muli %iv, %factor` (or
/// `%factor, %iv`) where %factor is defined outside the loop, group them by
/// factor, add one iter-arg per distinct factor, and replace the muli ops.
/// Returns true if the loop was changed.
static bool runMuliOnLoop(scf::ForOp loop, IRRewriter& rewriter) {
  Value iv = loop.getInductionVar();

  // Group muli ops by their loop-invariant factor, preserving insertion order.
  llvm::MapVector<Value, SmallVector<arith::MulIOp>> muls_by_factor;
  for (Operation* user : iv.getUsers()) {
    auto mul = dyn_cast<arith::MulIOp>(user);
    if (!mul) continue;
    Value factor = mul.getLhs() == iv ? mul.getRhs() : mul.getLhs();
    if (factor == iv) continue;  // iv * iv — skip
    if (!loop.isDefinedOutsideOfLoop(factor)) continue;
    auto& bucket = muls_by_factor[factor];
    if (!llvm::is_contained(bucket, mul)) bucket.push_back(mul);
  }
  if (muls_by_factor.empty()) return false;

  Location loc = loop.getLoc();
  SmallVector<Value> inits, increments;
  rewriter.setInsertionPoint(loop);
  for (auto& [factor, _] : muls_by_factor) {
    inits.push_back(rewriter.createOrFold<arith::MulIOp>(
        loc, loop.getLowerBound(), factor));
    increments.push_back(
        rewriter.createOrFold<arith::MulIOp>(loc, loop.getStep(), factor));
  }

  auto yield_fn = [&](OpBuilder& b, Location l,
                      ArrayRef<BlockArgument> new_args) -> SmallVector<Value> {
    SmallVector<Value> yields;
    for (auto [arg, inc] : llvm::zip(new_args, increments))
      yields.push_back(arith::AddIOp::create(b, l, arg, inc));
    return yields;
  };

  FailureOr<LoopLikeOpInterface> result = loop.replaceWithAdditionalYields(
      rewriter, inits, /*replaceInitOperandUsesInLoop=*/false, yield_fn);
  if (failed(result)) return false;

  auto new_loop = cast<scf::ForOp>(*result);
  auto new_iter_args =
      new_loop.getRegionIterArgs().take_back(muls_by_factor.size());
  for (auto [entry, iter_arg] : llvm::zip(muls_by_factor, new_iter_args))
    for (arith::MulIOp mul : entry.second) rewriter.replaceOp(mul, iter_arg);
  return true;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct LoopIVAffineMulStrengthReductionPass
    : public ktdf::impl::LoopIVAffineMulStrengthReductionPassBase<
          LoopIVAffineMulStrengthReductionPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;

    // -----------------------------------------------------------------------
    // Sub-transformation 1: affine.apply weighted-sum -> iter-arg recurrences.
    //
    // Collect apply ops upfront (stable pointers — they are never replaced,
    // only erased after replaceOp).  For each one we re-derive the
    // participating loop list from the *current* IR immediately before
    // transforming it.  This ensures we always hold fresh, live loop handles
    // even when a previous iteration's replaceWithAdditionalYields replaced
    // an enclosing loop that is also an ancestor of the current apply.
    // -----------------------------------------------------------------------
    SmallVector<affine::AffineApplyOp> applies;
    getOperation()->walk(
        [&](affine::AffineApplyOp op) { applies.push_back(op); });

    IRRewriter rewriter(&getContext());

    auto buildParticipating =
        [](affine::AffineApplyOp apply,
           SmallVectorImpl<ParticipatingLoop>& participating) -> bool {
      AffineMap map = apply.getAffineMap();
      if (map.getNumResults() != 1) return false;
      auto coeffs = matchWeightedDimSum(map.getResult(0));
      if (!coeffs || coeffs->empty()) return false;
      ValueRange operands = apply.getMapOperands();
      if (operands.size() != map.getNumDims()) return false;

      SmallVector<scf::ForOp> enclosing = enclosingLoops(apply);
      bool valid = true;
      for (auto& [dim_idx, coeff] : *coeffs) {
        if (dim_idx >= operands.size()) {
          valid = false;
          break;
        }
        Value iv_val = operands[dim_idx];
        int enc_idx = -1;
        for (auto [i, l] : llvm::enumerate(enclosing)) {
          if (l.getInductionVar() == iv_val) {
            enc_idx = (int)i;
            break;
          }
        }
        if (enc_idx == -1) {
          valid = false;
          break;
        }
        scf::ForOp l = enclosing[enc_idx];
        participating.push_back(
            {l, l.getLowerBound(), l.getStep(), coeff, enc_idx});
      }
      if (!valid || participating.size() < 2) return false;
      llvm::sort(participating,
                 [](const ParticipatingLoop& a, const ParticipatingLoop& b) {
                   return a.enclosing_index > b.enclosing_index;
                 });
      for (unsigned i = 1; i < participating.size(); ++i)
        if (participating[i].enclosing_index >=
            participating[i - 1].enclosing_index)
          return false;
      return true;
    };

    for (affine::AffineApplyOp apply : applies) {
      SmallVector<ParticipatingLoop> participating;
      if (!buildParticipating(apply, participating)) continue;
      if (transformCandidate(apply, participating, rewriter)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME << "] strength-reduced affine.apply over "
                   << participating.size() << " loops\n");
      }
    }

    // -----------------------------------------------------------------------
    // Sub-transformation 2: arith.muli %iv, %factor -> iter-arg recurrence.
    // Collect loops first; replaceWithAdditionalYields invalidates the handle
    // but moves the body intact, keeping nested loop handles valid.
    // -----------------------------------------------------------------------
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops) {
      if (runMuliOnLoop(loop, rewriter))
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME << "] strength-reduced arith.muli IV\n");
    }
  }
};

}  // namespace

auto ktdf::createLoopIVAffineMulStrengthReductionPass()
    -> std::unique_ptr<Pass> {
  return std::make_unique<LoopIVAffineMulStrengthReductionPass>();
}
