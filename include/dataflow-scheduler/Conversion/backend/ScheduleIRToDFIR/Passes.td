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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_SCHEDULEIRTODFIR_PASSES_TD
#define DATAFLOW_SCHEDULER_CONVERSION_SCHEDULEIRTODFIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def KTDFToKTDFLoweringPass : Pass<"ktdf-to-ktdflowering", "mlir::ModuleOp"> {
  let summary = "Lower KTDF pipelines/stages to KTDFLowering execute_on/signal";
  let description = [{
    Converts KTDF pipelines/stages with applicable_units attributes to
    KTDFLowering IR using execute_on and signal operations with operand-based
    unit selection.

    Phase 1.5: Operand Infrastructure
    - Component classification (parallel vs non-parallel)
    - Grid extraction from function attributes
    - Unit materialization (dataflow.get_unit operations)
    - Uniform infrastructure (maps and queries for dynamic unit selection)

    Phase 2: KTDF-to-KTDFLowering Transformation
    - Wire queries to stages based on applicable_units
    - Analyze stage dependencies via token flow
    - Compute scratchpad conflicts
    - Generate pipeline-level execute_on with all units
    - Generate nested stage-level execute_on with stage-specific units
    - Insert signal operations for conflict resolution
    - Remove old KTDF structures
  }];
  let constructor = "::scheduler::createKTDFToKTDFLoweringPass()";
  let dependentDialects = [
    "mlir::dataflow::DataflowDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::ktdf_lowering::KTDFLoweringDialect",
    "mlir::ktdp::KtdpDialect",
    "mlir::uniform::UniformDialect"
  ];
}

def KTDFLowToDFIRPass : Pass<"ktdflowering-to-dfir", "mlir::ModuleOp"> {
  let summary = "Create dataflow.program_unit ops from KTDFLowering execute_on/signal";
  let description = [{
    For each function, partitions top-level ops into prelude
    (dataflow.get_unit, arith.constant, uniform.def_immutable_mapping,
    uniform.query_map) and work; discovers component types in use; emits one
    dataflow.program_unit per type containing a self-contained clone of the
    work region with ktdf_lowering.execute_on ops unwrapped or dropped per
    type-applicability and ktdf_lowering.signal ops filtered similarly.
  }];
  let constructor = "::scheduler::createKTDFLowToDFIRPass()";
  let dependentDialects = [
    "mlir::agen::AgenDialect",
    "mlir::dataflow::DataflowDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::ktdf_lowering::KTDFLoweringDialect",
    "mlir::ktdp::KtdpDialect",
    "mlir::symbol::SymbolDialect",
    "mlir::uniform::UniformDialect"
  ];
}

def WrapProgramDFIRPass : Pass<"wrap-program-dfir", "mlir::ModuleOp"> {
  let summary = "Give each program two levels: what it is, and what it is compiled from";
  let description = [{
    A program of the run becomes a function of its own name and signature holding
    what it is -- which of the run's inputs it is about, and which symbol each of
    them is -- and an inner module holding a function of no arguments holding what
    it is compiled from.

    The symbol declarations move out of the DataflowIR into the outer function, so
    that what is handed to code generation needs nothing only a caller could
    supply. A program whose DataflowIR reads an argument for anything other than
    saying what a symbol is cannot be compiled at all, and is reported.

    Programs that already have two levels are left alone.
  }];
  let constructor = "::scheduler::createWrapProgramDFIRPass()";
  let dependentDialects = [
    "mlir::dataflow::DataflowDialect",
    "mlir::func::FuncDialect",
    "mlir::symbol::SymbolDialect"
  ];
}

def SplitDFIROutputPass : Pass<"emit-split-dfir", "mlir::ModuleOp"> {
  let summary = "Split hierarchical DFIR into flat per-function modules for DCC";
  let description = [{
    Converts the two-level hierarchical DFIR produced by KTDFLowToDFIRPass
    into a set of flat modules, one per function, written to disk:
      - global.mlir: declaration module (entry functions + forward decls)
      - <funcname>.mlir: one file per implementation function that contains
        dataflow.program_unit ops
    The first child module is the declaration module; every later child module
    (e.g. one per schedule, as produced by wrap-program-dfir) is searched for
    implementation functions, including in nested modules. Other top-level ops
    (e.g. ktdf_arch.device) are ignored.
    Top-level aliases (#map, #set) are copied into every output module.
  }];
  let constructor = "::scheduler::createSplitDFIROutputPass()";
  let options = [
    Option<"outputDir", "output-dir", "std::string", /*default=*/"\"\"",
           "Directory to write split DFIR files; defaults to input file directory">
  ];
}

#endif // DATAFLOW_SCHEDULER_CONVERSION_SCHEDULEIRTODFIR_PASSES_TD
