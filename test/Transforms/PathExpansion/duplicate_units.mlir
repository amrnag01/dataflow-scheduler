// RUN: dataflow-scheduler-opt --path-expansion --split-input-file \
// RUN:   --verify-diagnostics %s

// A pipeline can run only one stage on each unit. Path expansion rejects input
// where two stages end up on the same unit, whether the unit is pinned in the
// input or assigned by the planner.

// Two stages load from DDR, so both would run on MNILU.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_loads_one_compute(%a: memref<64xf16, "DDR">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:7 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %t0, %t1, %t2, %t3 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      // expected-note @below {{other stage on unit "MNILU"}}
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      // expected-error @below {{path-expansion: stage runs on unit "MNILU", which another stage of the pipeline already runs on}}
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#3, %p#4) depends_out(%p#5) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %y = ktdf.read_from_fifo %p#1 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
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

// Two stages load from DDR into the two slots of one fifo.allocate, so both
// would run on MNILU.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @shared_alloc(%a: memref<64xf16, "DDR">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:7 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f#0, %f#1, %f2, %t0, %t1, %t2, %t3 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      // expected-note @below {{other stage on unit "MNILU"}}
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      // expected-error @below {{path-expansion: stage runs on unit "MNILU", which another stage of the pipeline already runs on}}
      ktdf.stage depends_in(none) depends_out(%p#4) {
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#3, %p#4) depends_out(%p#5) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %y = ktdf.read_from_fifo %p#1 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
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

// Two stages store to DDR, so both would run on MNISU.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_stores(%a: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">, %d: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:6 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %t0, %t1, %t2 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#3) depends_out(%p#4) {
        %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %s = linalg.add ins(%x, %x : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) -> tensor<64xf16>
        ktdf.write_to_fifo %s, %p#1 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
        ktdf.write_to_fifo %s, %p#2 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      // expected-note @below {{other stage on unit "MNISU"}}
      ktdf.stage depends_in(%p#4) depends_out(none) {
        ktdf.data_transfer from %p#1 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
      // expected-error @below {{path-expansion: stage runs on unit "MNISU", which another stage of the pipeline already runs on}}
      ktdf.stage depends_in(%p#4) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64] to %d[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
    }
    return
  }
}

// -----

// A DDR -> L1 stage and a stage loading from DDR would both run on MNILU.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @chain_and_direct(%a: memref<64xf16, "DDR">, %l1: memref<64xf16, "L1">, %b: memref<64xf16, "DDR">, %c: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{path-expansion: failed to plan path expansion}}
    ktdf.pipeline {
      %p:8 = ktdf.private -> (!ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token
        %t4 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %f2, %t0, %t1, %t2, %t3, %t4 : !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      // expected-note @below {{other stage on unit "MNILU"}}
      ktdf.stage depends_in(none) depends_out(%p#3) {
        ktdf.data_transfer from %a[%c0] size [64] to %l1[%c0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
      }
      ktdf.stage depends_in(%p#3) depends_out(%p#4) {
        ktdf.data_transfer from %l1[%c0] size [64] to %p#0 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
      }
      // expected-error @below {{path-expansion: stage runs on unit "MNILU", which another stage of the pipeline already runs on}}
      ktdf.stage depends_in(none) depends_out(%p#5) {
        ktdf.data_transfer from %b[%c0] size [64] to %p#1 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }
      ktdf.stage depends_in(%p#4, %p#5) depends_out(%p#6) {
        %x = ktdf.read_from_fifo %p#0 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
        %y = ktdf.read_from_fifo %p#1 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf16>
        %s = linalg.add ins(%x, %y : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) -> tensor<64xf16>
        ktdf.write_to_fifo %s, %p#2 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#6) depends_out(none) {
        ktdf.data_transfer from %p#2 size [64] to %c[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      }
    }
    return
  }
}
