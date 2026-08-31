//===-- KTDPLoweringOps.cpp -------------------------------------*- C++ -*-===//
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
// This file implements the ktdp_lowering dialect operations.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
// clang-format on

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ktdp_lowering;

//===----------------------------------------------------------------------===//
// KTDPLoweringDialect — op registration
//===----------------------------------------------------------------------===//

void KTDPLoweringDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp — verifier
//===----------------------------------------------------------------------===//

LogicalResult ConstructIndirectAccessTileOp::verify() {
  // ind_addr_buf_subscripts count must equal the rank of ind_addr_buf_memref.
  auto iab_type = mlir::cast<mlir::MemRefType>(getIndAddrBufMemref().getType());
  auto num_subscripts = static_cast<int64_t>(getIndAddrBufSubscripts().size());
  if (num_subscripts != iab_type.getRank())
    return emitOpError() << "ind_addr_buf_subscripts has " << num_subscripts
                         << " operand(s) but ind_addr_buf_memref has rank "
                         << iab_type.getRank() << "; they must be equal";

  return success();
}
