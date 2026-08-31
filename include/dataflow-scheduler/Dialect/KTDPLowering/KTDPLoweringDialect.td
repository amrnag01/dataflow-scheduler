//===-- KTDPLoweringDialect.td -----------------------------*- tablegen -*-===//
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
// This file defines the ktdp_lowering dialect, which holds some lowered forms
// of ktdp operations that are produced during the scheduling transformation
// pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERINGDIALECT_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERINGDIALECT_TD

include "mlir/IR/OpBase.td"

def KTDPLowering_Dialect : Dialect {
  let name = "ktdp_lowering";
  let cppNamespace = "::mlir::ktdp_lowering";

  let summary = "Lowered form of some operations for the ktdp dialect";
  let description = [{
    The `ktdp_lowering` dialect holds lowered forms of `ktdp` operations.
    Operations in this dialect are temporary: they are introduced by conversion
    passes and consumed during that conversion phase in the pipeline.
  }];

  let extraClassDeclaration = [{
    void registerOps();
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDPLOWERING_KTDPLOWERINGDIALECT_TD
