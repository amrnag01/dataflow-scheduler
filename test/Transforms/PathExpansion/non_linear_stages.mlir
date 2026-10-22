// RUN: dataflow-scheduler-opt --path-expansion --split-input-file %s | FileCheck %s
// NOTE: FIFO memory-space names ("DDR", "SFU", "L1") must match the kind
// strings in sample_device.mlir, which the routing graph keys on.

// Pipelines whose stages branch and merge. Every dependency between two
// original stages is expanded on its own. A pipeline can run only one stage on
// each unit, so a hop through a unit that already runs a stage is folded into
// that stage instead of getting a new one.

// One load stage reads DDR and needs an L1 hop on L1LU. The other load stage
// already runs on L1LU, so the hop is folded into it instead of adding a
// second stage on L1LU.
// CHECK-LABEL: func.func @mixed_branches(
// CHECK:         %[[P:.+]]:9 = ktdf.private
// CHECK-NEXT:    memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:    memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[P]]#5) {
// CHECK-NEXT:    ktdf.data_transfer from %arg0[0] size [64] to %[[P]]#0[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#5) depends_out(%[[P]]#6) {
// CHECK-NEXT:    ktdf.data_transfer from %arg1[0] size [64] to %[[P]]#4 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    ktdf.data_transfer from %[[P]]#0[0] size [64] to %[[P]]#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["L1LU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#6) depends_out(%[[P]]#7) {
// CHECK-NEXT:    ktdf.read_from_fifo %[[P]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:    ktdf.read_from_fifo %[[P]]#4 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK:         ktdf.write_to_fifo %{{.*}}, %[[P]]#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["SFU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#7) depends_out(%[[P]]#8) {
// CHECK-NEXT:    ktdf.data_transfer from %[[P]]#2 size [64] to %[[P]]#3[0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["L1SU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#8) depends_out(none) {
// CHECK-NEXT:    ktdf.data_transfer from %[[P]]#3[0] size [64] to %arg2[0] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
// CHECK-NEXT:    } {applicable_units = ["MNISU"]}

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @mixed_branches(%a: memref<64xf16, "DDR">, %b: memref<64xf16, "L1">, %c: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %p:7 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %t0, %t1, %t2, %t3 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#3, %p#4) depends_out(%p#5) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %y = ktdf.read_from_fifo %p#1 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %s = linalg.add ins(%x, %y : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) -> tensor<64xf16>
        ktdf.write_to_fifo %s, %p#2 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#5) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
    }
    return
  }
}

// -----

// The compute stage writes two results. One is stored to DDR and needs an L1
// hop on L1SU; the other is stored to L1 by a stage that already runs on L1SU,
// so the hop is folded into that stage. The load needs an L1 hop on L1LU,
// which no stage runs on yet, so a new stage is added for it.
// CHECK-LABEL: func.func @store_fold(
// CHECK:         %[[P:.+]]:9 = ktdf.private
// CHECK-NEXT:    memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:    memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[P]]#5) {
// CHECK-NEXT:    ktdf.data_transfer from %arg0[0] size [64] to %[[P]]#0[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#5) depends_out(%[[P]]#6) {
// CHECK-NEXT:    ktdf.data_transfer from %[[P]]#0[0] size [64] to %[[P]]#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["L1LU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#6) depends_out(%[[P]]#7) {
// CHECK-NEXT:    ktdf.read_from_fifo %[[P]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK:         ktdf.write_to_fifo %{{.*}}, %[[P]]#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:    ktdf.write_to_fifo %{{.*}}, %[[P]]#4 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:    } {applicable_units = ["SFU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#7) depends_out(%[[P]]#8) {
// CHECK-NEXT:    ktdf.data_transfer from %[[P]]#4 size [64] to %arg2[0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
// CHECK-NEXT:    ktdf.data_transfer from %[[P]]#2 size [64] to %[[P]]#3[0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
// CHECK-NEXT:    } {applicable_units = ["L1SU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[P]]#8) depends_out(none) {
// CHECK-NEXT:    ktdf.data_transfer from %[[P]]#3[0] size [64] to %arg1[0] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
// CHECK-NEXT:    } {applicable_units = ["MNISU"]}

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @store_fold(%a: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">, %d: memref<64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %p:6 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %t0, %t1, %t2 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#3) depends_out(%p#4) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %s = linalg.add ins(%x, %x : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) -> tensor<64xf16>
        ktdf.write_to_fifo %s, %p#1 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
        ktdf.write_to_fifo %s, %p#2 : tensor<64xf16>, <"SFU" -> "L1", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#4) depends_out(none) {
        ktdf.data_transfer from %p#1 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
      ktdf.stage depends_in(%p#4) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64] to %d[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<64xf16, "L1">
      }
    }
    return
  }
}
