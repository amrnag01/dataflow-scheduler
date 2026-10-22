// RUN: dataflow-scheduler-opt --ktir-pipeline %s | FileCheck %s

// Reshapes between a FIFO and a compute are folded into the FIFO read/write,
// since a FIFO slot is flat.

// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-NOT:       tensor.collapse_shape
// CHECK-NOT:       tensor.expand_shape
// CHECK:           ktdf.data_transfer from %{{.*}} size [1, 1, 64] to %{{.*}} size [1, 1, 64]
// CHECK:           %[[READ:.*]] = ktdf.read_from_fifo %{{.*}} : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:      %[[EMPTY:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:      %[[GENERIC:.*]] = linalg.generic {{.*}} ins(%[[READ]] : tensor<64xf16>) outs(%[[EMPTY]] : tensor<64xf16>)
// CHECK:           } -> tensor<64xf16>
// CHECK-NEXT:      ktdf.write_to_fifo %[[GENERIC]], %{{.*}} : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
// CHECK:           ktdf.data_transfer from %{{.*}} size [1, 1, 64] to %{{.*}} size [1, 1, 64]
// CHECK-NOT:       tensor.collapse_shape
// CHECK-NOT:       tensor.expand_shape
// CHECK:           return

#id = affine_map<(d0) -> (d0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1(%src: memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">,
                              %dst: memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %load = ktdp_lowering.load %src[%i, 0, 0] [1, 1, 64] [1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR"> -> tensor<1x1x64xf16>
      %flat = tensor.collapse_shape %load [[0, 1, 2]] : tensor<1x1x64xf16> into tensor<64xf16>
      %empty = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%flat : tensor<64xf16>) outs(%empty : tensor<64xf16>)
          attrs = {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
      ^bb0(%in: f16, %out: f16):
        %sum = arith.addf %in, %in : f16
        linalg.yield %sum : f16
      } -> tensor<64xf16>
      %expanded = tensor.expand_shape %result [[0, 1, 2]] output_shape [1, 1, 64] : tensor<64xf16> into tensor<1x1x64xf16>
      ktdp_lowering.store %expanded into %dst[%i, 0, 0] [1, 1, 64] [1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x1x64xf16> into memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
