//===-- Passes.td ------------------------------------------*- tablegen -*-===//
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
// This file declares all the scheduler passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD
#define DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ComputeGroupExtractionPass : Pass<"compute-group-extraction", "mlir::ModuleOp"> {
  let summary = "Extract main and compute group modules from a module";
  let description = [{
    This pass extracts the main and compute groups from the initial module.
  }];
  let constructor = "::scheduler::createComputeGroupExtractionPass()";
}

def KTIRLegalityCheckPass : Pass<"ktir-legality-check", "mlir::ModuleOp"> {
  let summary = "Reject KTIR constructs V1 cannot lower yet";
  let description = [{
    Fails fast with a diagnostic when the input contains V1-unsupported
    constructs: scf.for with loop-carried args, scf.while, compute ops outside
    {add, mul, sub}, or the ktdp distributed-memory-view / indirect-access-tile
    ops. Mutates nothing; legal input passes through.
  }];
  let constructor = "::scheduler::createKTIRLegalityCheckPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def KTIRMapAndTilePass : Pass<"ktir-map-and-tile", "mlir::func::FuncOp"> {
  let summary = "Maps and tiles KTIR programs";
  let description = [{
    Performs device-specific mapping and tiling transformations on a normalized
    KTIR input program, leading to parallel loop nests with their load/store
    operations having been sunk/hoisted.
  }];
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect",
    "mlir::ktdf_arch::KTDFArchDialect",
    "mlir::ktdp_lowering::KTDPLoweringDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def KTIRBufferizePass : Pass<"ktir-bufferize", "mlir::func::FuncOp"> {
  let summary = "Converts value-like accesses to memory load/store";
  let description = [{
    Performs lowering transformations that replace tensors and access tiles
    with actual backing memory and `memref`s where needed.

    The following ops are bufferized by this pass:
      - `ktdp.(load|store)`
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def KTIRPipelinePass : Pass<"ktir-pipeline", "mlir::func::FuncOp"> {
  let summary = "Constructs pipelines in bufferized KTIR programs";
  let description = [{
    Detects all top-level loop nests and attempts to construct a pipeline
    within them. The pipeline will be initialized with all the 
    `ktdp_lowering.store` found inside the loop body, growing by inserting
    producers inside the loop body while possible.

    After pipeline creation, lowering and legalization patterns are applied 
    until the pipeline is normal form. The pipeline normal form is defined as
    follows:

    ```mlir
    ktdf.pipeline {
      %... = ktdf.private {
        // tokens...
        // FIFOs...
      }

      // Load stages (1 per unique memory space)
      ktdf.stage depends_in(none) depends_out(%tok0) {
        ktdf.data_transfer from %input to %in_fifo
      }

      // Compute stages (1 per unique mapping)
      ktdf.stage depends_in(%tok0) depends_out(%tok1) {
        %in = ktdf.read_from_fifo %in_fifo
        %out = compute %in
        ktdf.write_to_fifo %out, %out_fifo
      } {applicable_units = ["COMPUTE"]}

      // Store stages (1 per unique memory space)
      ktdf.stage depends_in(%tok1) depends_out(none) {
        ktdf.data_transfer from %out_fifo to %output
      }
    }
    ```

    In case that there were `ktdf.via` hops between two ops in the pipeline,
    these will materialize as follows (between the adjacent stages):

    ```mlir
    ktdf.pipeline {
      %... = ktdf.private {
        // ...
        // allocs...
      }

      // ...

      // Hop stages (2 per memory space) ...
      ktdf.stage depends_in(%tok0) depends_out(%tok1) {
        ktdf.data_transfer from %in_fifo to %hop_alloc
      }
      ktdf.stage depends_in(%tok1) depends_out(%tok2) {
        ktdf.data_transfer from %hop_alloc to %out_fifo
      }

      // ...
    }
    ```
  }];
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect",
    "mlir::ktdf_arch::KTDFArchDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def IndirectComputeGroupSplitPass
    : Pass<"indirect-compute-group-split", "mlir::ModuleOp"> {
  let summary = "Split indirect-access compute groups into addr-resolve and "
                "gather/scatter subgroups";
  let description = [{
    Finds every extracted child-module function that contains a
    `ktdp.construct_indirect_access_tile` op and splits it into two sibling
    child modules: one that resolves raw integer indices to flat absolute
    addresses and stores them into a global address buffer, and one that loads
    the address buffer into the ind_addr_buf (IAB) and performs the actual
    gather/scatter compute using `ktdp_lowering.construct_indirect_access_tile`.

    Both gather (indirect load) and scatter (indirect store) are handled.
    The pass verifies that each extracted child-module function contains at
    most one `ktdp.construct_indirect_access_tile` and fails with a diagnostic
    if this invariant is violated.

    Child-module functions that do not contain an indirect op are left
    unchanged. This pass is a no-op on the orchestrator module.
  }];
  let constructor = "::scheduler::createIndirectComputeGroupSplitPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::ktdp::KtdpDialect",
    "mlir::ktdp_lowering::KTDPLoweringDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::spyreop::SpyreOpDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def FuseLinalgPass : Pass<"fuse-linalg", "mlir::func::FuncOp"> {
  let summary = "Fuses `linalg.generic` operations";
  let description = [{
    Performs mapping-aware fusion of `linalg.generic` operations. Currently,
    only elementiwse operations are considered for fusion.

    Mapping awareness means that `linalg.generic` operations which are mapped,
    or imply a mapping based on their contents, will only be fused with other
    ops that have compatible mappings. Mappings are compatible if they refer to
    the exact same resources.

    As a result of this pass, implied mappings will be propagated upwards
    directly onto the `linalg.generic` operations that have them.
  }];
  let dependentDialects = [
    "mlir::ktdf_arch::KTDFArchDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD
