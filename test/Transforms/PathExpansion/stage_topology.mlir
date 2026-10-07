// Check the stage graph shapes the planner rejects. Every stage must be
// clearly before (load side) or after (store side) the compute stages.
// Pipelines whose stages branch and merge are tested in
// non_linear_stages.mlir.

// RUN: not dataflow-scheduler-opt --path-expansion --split-input-file \
// RUN:   --debug-only=path-expansion-planner %s 2>&1 | FileCheck %s

// A stage that neither feeds nor is fed by a compute stage.

// CHECK:     Stage 3 is not connected to any compute stage
// CHECK:     error: path-expansion: failed to plan path expansion

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @unconnected_stage(%a: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">, %d: memref<64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %p:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1, %t2 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        ktdf.write_to_fifo %x, %p#1 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(none) {
        ktdf.data_transfer from %p#1 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.data_transfer from %a[%c0] size [64] to %d[%c0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
      }
    }
    return
  }
}

// -----

// A stage between two compute stages: it is fed by the first and feeds the
// second, so it is on neither side.

// CHECK:     Stage 2 sits between two compute stages
// CHECK:     error: path-expansion: failed to plan path expansion

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @stage_between_computes(%a: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">, %tmp: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %p:9 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token
        %t4 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %f3, %t0, %t1, %t2, %t3, %t4 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#4) depends_out(%p#5) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        ktdf.write_to_fifo %x, %p#1 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#5) depends_out(%p#6) {
        ktdf.data_transfer from %p#1 size [64] to %tmp[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
        ktdf.data_transfer from %tmp[%c0] size [64] to %p#2 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#6) depends_out(%p#7) {
        %y = ktdf.read_from_fifo %p#2 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        ktdf.write_to_fifo %y, %p#3 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#7) depends_out(none) {
        ktdf.data_transfer from %p#3 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
    }
    return
  }
}

// -----

// The only pinned stage is on a load/store unit, not a compute unit, so the
// pipeline has no compute stage to anchor the two sides.

// CHECK:     Pipeline has no stage pinned to a compute unit
// CHECK:     error: path-expansion: failed to plan path expansion

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @no_compute_stage(%a: memref<64xf16, "DDR">, %d: memref<64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.data_transfer from %a[%c0] size [64] to %d[%c0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
      } {applicable_units = ["MNILU"]}
    }
    return
  }
}
