//===-- ReductionDimChunking.cpp --------------------------------*- c++ -*-===//
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
// ReductionDimChunking: split reduction dimensions of the inner linalg.generic
// into sequential chunks so that each chunk fits in the hardware FIFO path.
//
// When the chunk count is inferred from the threshold (--num-chunks not given),
// chunking uses a GCD-based outer-to-inner strategy: starting from the
// outermost reduction dimension, each step absorbs gcd(dim_size, remaining)
// into that dimension and passes remaining/gcd inward.  Dimensions with a
// symbolic (dynamic) size are skipped.  The innermost overall loop dimension
// is never chunked — if it is a reduction dim it is excluded before
// distribution begins.
//
// When --num-chunks is given explicitly, multi-dim chunking is applied using
// one value per reduction dimension (in iterator-type order); one nested
// scf.for is emitted per dim; dims with chunk count 1 produce no loop.
//
// The pass operates on the shape produced by StageCoarsening.  It expects a
// top-level ktdf.pipeline with three sibling stages:
//
//   stage-0 (Load)   : global memory → local memory
//   stage-1 (Compute): outer scf.for (batch) whose body is a single inner
//                       ktdf.pipeline with three stages:
//                         stage-a (Load)   : local memory → FIFO
//                         stage-b (Compute): linalg.generic reduce
//                         stage-c (Store)  : FIFO → local memory
//   stage-2 (Store)  : local memory → global memory
//
// The transformation replaces the single inner ktdf.pipeline with nested
// scf.for loops — one per reduction dimension that has more than one chunk.
// Dimensions whose num_chunks==1 produce no loop (the single chunk covers
// the entire dimension).  Each innermost iteration contains one ktdf.pipeline
// with three stages built by ktdf::StageFactory (see ReductionUtils.h).
// First-vs-rest accumulation behaviour is selected at runtime via
// %condition = AND(iv_j == 0 for all active chunk loop IVs):
//
//   Load stage   : transfers the current input chunk slice (memref → fifo_in).
//                  When !condition, also transfers the partial accumulator
//                  (local memory output buffer → fifo_partial) so the
//                  Compute stage can read it back.
//   Compute stage: unconditionally reads fifo_in and runs a first
//                  linalg.generic into tensor.empty.  Then branches on
//                  condition:
//                    then (first iter): yield first generic result.
//                    else (later iters): read fifo_partial, run a second
//                      linalg.generic(input=partial, output=first result),
//                      yield second generic result.
//                  The conditional result is written to fifo_out.
//   Store stage  : unconditionally writes fifo_out back to the local memory
//                  output buffer.
//
// The existing local memory output buffer (discovered via the original Store
// stage's data_transfer destination) is reused as the partial-accumulation
// buffer; no new memref is allocated.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionChunkAnalysis.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "reduction-dim-chunking"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;
using namespace mlir::ktdf;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONDIMCHUNKINGPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Reduction Dim Chunking pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// Helper: find the unique linalg.generic with a reduction iterator inside a
// ktdf.stage body.  Returns nullptr if not found.
// ---------------------------------------------------------------------------
static linalg::GenericOp findReductionGenericOp(ktdf::StageOp stage) {
  linalg::GenericOp found;
  stage.getBody()->walk([&](linalg::GenericOp generic) {
    for (auto it : generic.getIteratorTypesArray()) {
      if (it == utils::IteratorType::reduction) {
        found = generic;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

// ---------------------------------------------------------------------------
// Pass implementation
// ---------------------------------------------------------------------------
struct ReductionDimChunkingPass
    : public ktdf::impl::ReductionDimChunkingPassBase<
          ReductionDimChunkingPass> {
  using ReductionDimChunkingPassBase<
      ReductionDimChunkingPass>::ReductionDimChunkingPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    ModuleOp module = getOperation();
    if (failed(transformModule(module))) signalPassFailure();
  }

 private:
  // -----------------------------------------------------------------------
  // Walk the module for linalg.generic ops that have at least one reduction
  // iterator.  For each such op, locate the immediately enclosing ktdf.stage
  // (the Compute stage) and collect unique stages into a list.  The list is
  // processed after the walk to avoid mutating IR while walking.
  // -----------------------------------------------------------------------
  LogicalResult transformModule(ModuleOp module) {
    // Use a DenseSet to avoid O(n²) duplicate checks.
    llvm::DenseSet<ktdf::StageOp> seen;
    SmallVector<ktdf::StageOp> compute_stages;
    module.walk([&](linalg::GenericOp generic) {
      // Check if this generic has a reduction iterator.
      bool has_reduction = false;
      for (auto it : generic.getIteratorTypesArray()) {
        if (it == utils::IteratorType::reduction) {
          has_reduction = true;
          break;
        }
      }
      if (!has_reduction) return WalkResult::advance();

      // linalg.generic is always directly nested inside a ktdf.stage
      // (structural invariant enforced by StageCoarsening).
      auto stage = generic->getParentOfType<ktdf::StageOp>();
      assert(stage && "linalg.generic not inside a ktdf.stage");

      if (seen.insert(stage).second) compute_stages.push_back(stage);
      return WalkResult::advance();
    });

    for (auto compute_stage : compute_stages)
      if (failed(transformPipeline(compute_stage))) return failure();

    return success();
  }

  // -----------------------------------------------------------------------
  // Entry point for one Compute stage.  Determines the reduction dims, chunk
  // counts, and per-dim chunk sizes, then delegates to rewriteComputeStage().
  // -----------------------------------------------------------------------
  LogicalResult transformPipeline(ktdf::StageOp compute_stage) {
    // Stages are always direct children of a pipeline.
    auto inner_pipeline = cast<ktdf::PipelineOp>(compute_stage->getParentOp());

    // Verify there is exactly one inner pipeline inside the parent region.
    unsigned inner_pipeline_count = 0;
    inner_pipeline->getParentRegion()->walk<WalkOrder::PreOrder>(
        [&](ktdf::PipelineOp) {
          ++inner_pipeline_count;
          return WalkResult::skip();
        });
    assert(inner_pipeline_count == 1 &&
           "expected exactly 1 inner pipeline in parent region");

    ktdf::StageOp load_stage =
        StageFactory::findLoadStage(inner_pipeline, compute_stage);
    assert(load_stage && "could not find load stage upstream of compute stage");

    ktdf::StageOp store_stage =
        StageFactory::findStoreStage(inner_pipeline, compute_stage);
    assert(store_stage &&
           "could not find store stage downstream of compute stage");

    // Locate the linalg.generic with a reduction iterator.
    linalg::GenericOp generic_op = findReductionGenericOp(compute_stage);
    assert(generic_op && "no reduction linalg.generic in compute stage");

    // FIXME: Current implementation leads to invalid IR when processing
    //        contractions, so skip them for now to allow matmul to pass.
    if (linalg::isaContractionOpInterface(generic_op)) {
      LDBG() << "ignoring contraction";
      return success();
    }

    // ------------------------------------------------------------------
    // Determine reduction dims and per-dim chunk sizes.
    //
    // Threshold path (numChunks empty): GCD-based outer-to-inner chunking
    // across all reduction dims; dynamic dims are skipped; the innermost
    // overall loop dim is excluded if it is a reduction dim.
    //
    // Explicit path (numChunks non-empty): one value per reduction dim;
    // nested scf.for loops; dims with count 1 produce no loop.
    // ------------------------------------------------------------------
    ReductionChunkResult reduction_chunk;

    // FIXME: This implementation assumes that there is an injective mapping
    //        from loop to shape space, i.e., for each iteration, there will be
    //        a distinct element read from the first input tensor.
    generic_op.getReductionDims(reduction_chunk.reduction_dims);

    if (numChunks.empty()) {
      // Auto-infer via analysis — GCD-based outer-to-inner chunking.

      // Collect all reduction dims, excluding the innermost overall loop dim
      // if it is a reduction (the innermost loop dim must never be chunked,
      // regardless of iterator type).
      if (!reduction_chunk.reduction_dims.empty() &&
          reduction_chunk.reduction_dims.back() ==
              generic_op.getNumLoops() - 1) {
        reduction_chunk.reduction_dims.pop_back();
      }
      if (reduction_chunk.reduction_dims.empty()) {
        LDBG(1) << PASS_NAME
            ": could not find a chunkable reduction dimension — skipping";
        return success();
      }

      auto result = analyzeReductionChunks(generic_op, chunkSizeThreshold);
      if (!result) {
        inner_pipeline.emitError(PASS_NAME
                                 ": ReductionChunkAnalysis could not infer "
                                 "chunk count");
        return failure();
      }

      auto chunk_dims =
          computeChunkDims(result->red_dim_sizes, result->num_chunks);
      if (!chunk_dims) {
        inner_pipeline.emitError(
            PASS_NAME
            ": chunk count could not be fully distributed across the "
            "chunkable reduction dimensions");
        return failure();
      }

      reduction_chunk.num_chunks = result->num_chunks;
      reduction_chunk.red_dim_sizes = std::move(result->red_dim_sizes);
      reduction_chunk.chunk_sizes = std::move(*chunk_dims);
    } else {
      // User-supplied --num-chunks path: collect reduction dims and validate.
      if (reduction_chunk.reduction_dims.empty()) {
        inner_pipeline.emitError(PASS_NAME
                                 ": could not find reduction dimension");
        return failure();
      }

      if (numChunks.size() != reduction_chunk.reduction_dims.size()) {
        inner_pipeline.emitError(
            llvm::Twine(PASS_NAME ": numChunks has ") +
            llvm::Twine(numChunks.size()) + " entries but there are " +
            llvm::Twine(reduction_chunk.reduction_dims.size()) +
            " reduction dims");
        return failure();
      }

      const auto loop_ranges = generic_op.getStaticLoopRanges();
      for (size_t j = 0; j < reduction_chunk.reduction_dims.size(); ++j) {
        const auto dim = reduction_chunk.reduction_dims[j];
        const auto dim_size = loop_ranges[dim];
        reduction_chunk.red_dim_sizes.push_back(dim_size);
        if (dim_size == ShapedType::kDynamic) {
          LDBG(1) << PASS_NAME
                  << ": dynamic reduction size not yet supported — skipping";
          return success();
        }
        unsigned nchunk = numChunks[j];
        if (nchunk == 0 || dim_size % nchunk != 0) {
          LDBG(1) << PASS_NAME ": num_chunks=" << nchunk
                  << " does not evenly divide dim " << dim
                  << " size=" << dim_size << " — skipping";
          return success();
        }
        reduction_chunk.chunk_sizes.push_back(dim_size /
                                              static_cast<int64_t>(nchunk));
      }
    }

    // Collect per-dim chunk counts in reduction-dim order.
    SmallVector<int64_t> per_dim_num_chunks;
    if (numChunks.empty()) {
      // Auto-inferred path: derive per-dim count from dim_size / chunk_size.
      for (size_t j = 0; j < reduction_chunk.reduction_dims.size(); ++j) {
        const int64_t dim_size = reduction_chunk.red_dim_sizes[j];
        // Dynamic dims were skipped by computeChunkDims (chunk_sizes[j] left
        // at kDynamic) — treat as 1 chunk (no loop emitted).
        if (dim_size == ShapedType::kDynamic)
          per_dim_num_chunks.push_back(1);
        else
          per_dim_num_chunks.push_back(dim_size /
                                       reduction_chunk.chunk_sizes[j]);
      }
    } else {
      for (unsigned nc : numChunks)
        per_dim_num_chunks.push_back(static_cast<int64_t>(nc));
    }

    LDBG(1) << PASS_NAME ": num_reduction_dims="
            << reduction_chunk.reduction_dims.size();

    // One chunk on every dimension leaves the reduction as it is: the rewrite
    // below would rebuild the pipeline to emit the same program, differing only
    // in how the indices print. It also keeps the pass off a single-level
    // pipeline, which has no batch loop for rewriteComputeStage to index
    // against.
    if (llvm::all_of(per_dim_num_chunks,
                     [](int64_t chunks) { return chunks == 1; })) {
      LDBG(1) << PASS_NAME ": every reduction dim has one chunk, nothing to do";
      return success();
    }

    return rewriteComputeStage(inner_pipeline, load_stage, compute_stage,
                               store_stage, generic_op,
                               reduction_chunk.reduction_dims,
                               reduction_chunk.chunk_sizes, per_dim_num_chunks);
  }

  // -----------------------------------------------------------------------
  // Replace inner_pipeline with nested scf.for loops — one per reduction
  // dimension whose num_chunks > 1. Dimensions with num_chunks == 1 need no
  // loop; their IV is treated as the constant 0 for offset and condition
  // computation.
  //
  // For N dims with num_chunks[j] > 1 the emitted structure is:
  //
  //   scf.for %iv_0 = 0 to num_chunks[0] step 1 {
  //     scf.for %iv_1 = 0 to num_chunks[1] step 1 {
  //       ...
  //         %condition = (iv_0 == 0 && iv_1 == 0 && ...)
  //         ktdf.pipeline {
  //           ktdf.private { ... }   // FIFO slots + tokens
  //           ktdf.stage { ... }     // Load
  //           ktdf.stage { ... }     // Compute
  //           ktdf.stage { ... }     // Store
  //         }
  //     }
  //   }
  //
  // The original inner_pipeline is erased after the replacement is inserted.
  // -----------------------------------------------------------------------
  LogicalResult rewriteComputeStage(
      ktdf::PipelineOp inner_pipeline, ktdf::StageOp load_stage,
      ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
      linalg::GenericOp generic_op, ArrayRef<unsigned> reduction_dims,
      ArrayRef<int64_t> chunk_sizes, ArrayRef<int64_t> per_dim_num_chunks) {
    MLIRContext* context = inner_pipeline.getContext();
    IRRewriter rewriter(context);
    Location loc = inner_pipeline.getLoc();

    // ------------------------------------------------------------------
    // Step 1: Trace dataflow to discover the source/destination memrefs and
    // the original FIFO slot types.
    //
    // Input path:
    //   input_memref → DataTransferOp → fifo_in → ReadFromFifoOp
    //               → generic.ins[0]
    //
    // Output path:
    //   generic.result(0) → WriteToFifoOp → fifo_out
    //               → DataTransferOp → partial_memref
    // ------------------------------------------------------------------
    auto [load_transfer, read_from_fifo] =
        StageFactory::findLoadTransfer(load_stage, generic_op);
    if (!read_from_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic input is not produced by read_from_fifo");
      return failure();
    }
    if (!load_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no memref-to-fifo transfer feeding "
                               "the generic's fifo_in slot in load stage");
      return failure();
    }
    Value input_memref = load_transfer.getSource();

    auto [store_transfer, write_to_fifo] =
        StageFactory::findStoreTransfer(compute_stage, store_stage, generic_op);
    if (!write_to_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic result is not consumed by write_to_fifo");
      return failure();
    }
    if (!store_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no fifo-to-memref transfer consuming "
                               "the generic's fifo_out slot in store stage");
      return failure();
    }
    Value partial_memref = store_transfer.getDestination();
    auto output_memref_type = cast<MemRefType>(partial_memref.getType());

    auto in_fifo_type =
        dyn_cast<ktdf::FifoSlotType>(read_from_fifo.getFifoSlot().getType());
    auto out_fifo_type =
        dyn_cast<ktdf::FifoSlotType>(write_to_fifo.getFifoSlot().getType());
    if (!in_fifo_type || !out_fifo_type) {
      inner_pipeline.emitError(
          PASS_NAME
          ": could not derive input/output FIFO slot types "
          "from compute stage");
      return failure();
    }

    // Per-chunk input tensor type: reduction dims replaced with chunk_sizes.
    auto orig_input_tensor_type =
        cast<RankedTensorType>(generic_op.getInputs().front().getType());
    SmallVector<int64_t> chunk_input_shape(
        orig_input_tensor_type.getShape().begin(),
        orig_input_tensor_type.getShape().end());
    for (size_t j = 0; j < reduction_dims.size(); ++j)
      chunk_input_shape[reduction_dims[j]] = chunk_sizes[j];
    auto chunk_input_tensor_type = RankedTensorType::get(
        chunk_input_shape, orig_input_tensor_type.getElementType());

    // Output tensor type (unchanged).
    auto output_tensor_type =
        cast<RankedTensorType>(generic_op.getOutputs().front().getType());

    // ------------------------------------------------------------------
    // Step 2: Emit the replacement structure.
    //
    // Insertion point is set just before inner_pipeline so the new loops land
    // in the correct position.  inner_pipeline is erased at the end (Step 3).
    // ------------------------------------------------------------------
    rewriter.setInsertionPoint(inner_pipeline);

    auto input_memref_type = cast<MemRefType>(input_memref.getType());
    int64_t chunk_fifo_elements = chunk_input_tensor_type.getNumElements();

    auto chunk_in_fifo_type = ktdf::FifoSlotType::get(
        context, in_fifo_type.getSrc(), in_fifo_type.getDest(),
        chunk_fifo_elements, in_fifo_type.getElementType());

    // Applicable-units attributes from original stages.
    auto load_units = load_stage.getApplicableUnitsAttr();
    auto compute_units = compute_stage.getApplicableUnitsAttr();
    auto store_units = store_stage.getApplicableUnitsAttr();

    // Batch-loop induction variable. The enclosing loop is a precondition:
    // the indices of the transfers built below are expressed relative to it.
    // A single-level pipeline is well-formed IR, so report the missing loop
    // rather than assert on it.
    auto batch_for =
        dyn_cast<scf::ForOp>(inner_pipeline->getParentRegion()->getParentOp());
    if (!batch_for)
      return inner_pipeline.emitError(PASS_NAME)
             << ": reduction needs chunking, but its pipeline is not nested in "
                "a batch scf.for, so there is no index to chunk against";
    Value batch_iv = batch_for.getInductionVar();

    // Materialise per-dim upper-bound constants before the outermost loop.
    // These are index-typed scf.for bounds and must live at the scf.for
    // scope, not inside any stage body.
    size_t n_dims = reduction_dims.size();
    SmallVector<Value> c_per_dim_num_chunks;
    for (int64_t nc : per_dim_num_chunks) {
      c_per_dim_num_chunks.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, nc));
    }

    // Build loops from outermost (dim 0) to innermost (dim n_dims-1).
    // dim_ivs[j] holds the loop IV when a loop was created for dim j, or
    // c0_loop when num_chunks[j]==1 (no loop for that dim).
    // After this block, inner_builder is positioned at the innermost loop
    // body (or at the original insertion point if no loops were generated).
    SmallVector<Value> dim_ivs(n_dims);
    OpBuilder inner_builder = rewriter;  // copy: same insertion point

    // c0_loop / c1_loop: loop-bound constants emitted at scf.for scope.
    // StageFactory emits its own c0/c1 inside each stage body where they
    // are legal (stage bodies are the only valid ktdf.pipeline children).
    Value c0_loop = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1_loop = arith::ConstantIndexOp::create(rewriter, loc, 1);

    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) {
        // No loop for this dim; use constant 0 as the IV.
        dim_ivs[j] = c0_loop;
      } else {
        auto loop = scf::ForOp::create(inner_builder, loc, c0_loop,
                                       c_per_dim_num_chunks[j], c1_loop);
        dim_ivs[j] = loop.getInductionVar();
        // Descend into the loop body for the next (inner) level.
        inner_builder = OpBuilder(loop.getBody(), loop.getBody()->begin());
      }
    }

    // condition = AND of (iv_j == 0) for every dim whose loop was generated.
    // Dims with num_chunks==1 always have iv==c0 and are skipped to avoid
    // dead arith.cmpi constants.
    // Degenerate case (all dims have num_chunks==1): condition stays null;
    // emit arith.constant 1 (i1) rather than a trivially-true cmpi.
    Value condition;
    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) continue;
      Value eq = arith::CmpIOp::create(
          inner_builder, loc, arith::CmpIPredicate::eq, dim_ivs[j], c0_loop);
      condition = condition
                      ? arith::AndIOp::create(inner_builder, loc, condition, eq)
                      : eq;
    }
    if (!condition) {
      condition = arith::ConstantIntOp::create(inner_builder, loc, /*value=*/1,
                                               /*width=*/1);
    }

    // One ktdf.pipeline per innermost chunk iteration.
    auto phase_pipeline = ktdf::PipelineOp::create(inner_builder, loc);
    OpBuilder pipe_bldr(phase_pipeline.getBody(),
                        phase_pipeline.getBody()->end());

    // partial_fifo_type carries output-tensor-sized elements in the
    // Load → Compute direction (same src/dest endpoints as fifo_in).
    auto partial_fifo_type = ktdf::FifoSlotType::get(
        context, in_fifo_type.getSrc(), in_fifo_type.getDest(),
        out_fifo_type.getNumElements(), out_fifo_type.getElementType());

    StageFactory factory(pipe_bldr, loc, batch_iv, load_units, compute_units,
                         store_units);

    // Describe the FIFO slot layout and token count for this chunk pipeline:
    //   in-slot  0  (chunk_in_fifo_type) : input chunk,  Load → Compute
    //   partial-slot 0 (partial_fifo_type): partial accum, Load → Compute
    //                                       (non-first iterations only)
    //   out-slot 0  (out_fifo_type)      : reduction result, Compute → Store
    //   token 0                          : Load signals Compute
    //   token 1                          : Compute signals Store
    ChunkPipelineConfig cfg;
    cfg.in_slot_types = {chunk_in_fifo_type};
    cfg.partial_slot_types = {partial_fifo_type};
    cfg.out_slot_types = {out_fifo_type};
    cfg.n_tokens = 2;

    factory.setSlots(factory.buildFifoPrivate(cfg));

    factory.buildLoadStage(condition, input_memref, input_memref_type,
                           partial_memref, output_memref_type, reduction_dims,
                           chunk_sizes, dim_ivs, cfg);

    factory.buildComputeStage(condition, chunk_input_tensor_type,
                              output_tensor_type, generic_op, cfg);

    factory.buildStoreStage(partial_memref, cfg);

    // ------------------------------------------------------------------
    // Step 3: Erase the original inner pipeline now that the replacement
    // has been inserted before it.
    // ------------------------------------------------------------------
    rewriter.eraseOp(inner_pipeline);

    return success();
  }
};

}  // namespace

auto mlir::ktdf::createReductionDimChunkingPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>();
}
