// RUN: dataflow-scheduler-opt --ktir-pipeline %s | FileCheck %s

// Tests that reshapes of the data exchanged through FIFOs are folded into the
// FIFO reads and writes, and thus into the data transfers.

#id = affine_map<(d0) -> (d0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

// A reshape before a FIFO write is folded into the write.
// CHECK-LABEL: func.func @fold_into_fifo_write
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.data_transfer from %{{.+}}[0, 0] size [1, 64] to %{{.+}} size [1, 64]
// CHECK:           }
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.read_from_fifo {{.+}} -> tensor<64xf16>
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     tensor.collapse_shape
func.func @fold_into_fifo_write(%in: memref<1x64xf16, "DDR">, %out: memref<64xf16, "DDR">) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %load = ktdp_lowering.load %in[0, 0] [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "DDR"> -> tensor<1x64xf16>
    %collapsed = tensor.collapse_shape %load [[0, 1]] {ktdf_arch.maps_to = "DDR"} : tensor<1x64xf16> into tensor<64xf16>
    %init = tensor.empty() : tensor<64xf16>
    %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%collapsed : tensor<64xf16>) outs(%init : tensor<64xf16>) attrs = {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
    ^bb0(%x: f16, %o: f16):
      %y = spyreop.exp %x : f16
      linalg.yield %y : f16
    } -> tensor<64xf16>
    ktdp_lowering.store %result into %out[0] [64] [1] {dataflow_scheduler.throttle = 64 : i64} : tensor<64xf16> into memref<64xf16, "DDR">
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}

// A reshape after a FIFO read is folded into the read.
// CHECK-LABEL: func.func @fold_into_fifo_read
// CHECK-NOT:     tensor.expand_shape
// CHECK:         ktdf.pipeline {
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.write_to_fifo {{.+}} : tensor<64xf16>
// CHECK:           }
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             ktdf.data_transfer from %{{.+}} size [1, 64] to %{{.+}}[0, 0] size [1, 64]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     tensor.expand_shape
func.func @fold_into_fifo_read(%in: memref<64xf16, "DDR">, %out: memref<1x64xf16, "DDR">) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %load = ktdp_lowering.load %in[0] [64] [1] {dataflow_scheduler.throttle = 64 : i64} : memref<64xf16, "DDR"> -> tensor<64xf16>
    %init = tensor.empty() : tensor<64xf16>
    %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%load : tensor<64xf16>) outs(%init : tensor<64xf16>) attrs = {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
    ^bb0(%x: f16, %o: f16):
      %y = spyreop.exp %x : f16
      linalg.yield %y : f16
    } -> tensor<64xf16>
    %expanded = tensor.expand_shape %result [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
    ktdp_lowering.store %expanded into %out[0, 0] [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x64xf16> into memref<1x64xf16, "DDR">
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
