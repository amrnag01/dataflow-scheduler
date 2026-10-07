// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s

// The FIFO between the L1 -> SFU load stage and the SFU compute stage is not
// adjacent to any synthetic stage, but both of its ends get an applicable unit
// during expansion. Its type must change from the memory-level "L1" -> "SFU"
// to "L1LU" -> "SFU" on the allocation, the producer and the consumer.

// CHECK-LABEL: func.func @retype_fifo_between_original_stages(
// CHECK-NOT:     "L1" -> "SFU"
// CHECK:         ktdf.private
// CHECK:           ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:           ktdf.private_yield
// CHECK:         ktdf.stage
// CHECK:           ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}] size [1, 64] to %{{.*}}[0, 0] size [1, 64]
// CHECK:         } {applicable_units = ["MNILU"]}
// CHECK:         ktdf.stage
// CHECK:           ktdf.data_transfer from %{{.*}}[0, 0] size [1, 64] to %[[FIFO:.*]] size [1, 64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:         } {applicable_units = ["L1LU"]}
// CHECK:         ktdf.stage
// CHECK:           ktdf.read_from_fifo %[[FIFO]] : <"L1LU" -> "SFU", 64xf16>
// CHECK:           ktdf.write_to_fifo %{{.*}}, %{{.*}} : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK:         } {applicable_units = ["SFU"]}
// CHECK:         ktdf.stage
// CHECK:         } {applicable_units = ["L1SU"]}
// CHECK:         ktdf.stage
// CHECK:         } {applicable_units = ["MNISU"]}
// CHECK-NOT:     "L1" -> "SFU"

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @retype_fifo_between_original_stages(%arg0: index) {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1024 = arith.constant 1024 : index
    %c12288 = arith.constant 12288 : index
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.muli %0, %c3 : index
    %2 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
    %3 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
    %4 = arith.addi %1, %arg0 : index
    %5 = arith.muli %4, %c64 : index
    %memspacecast = memref.memory_space_cast %2 : memref<96x64xf16> to memref<96x64xf16, "DDR">
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%5], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    %6 = arith.muli %4, %c64 : index
    %memspacecast_0 = memref.memory_space_cast %3 : memref<96x64xf16> to memref<96x64xf16, "DDR">
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [%6], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf.pipeline {
          %7:8 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">) {
            %8 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
            %9 = ktdf.create_token : !ktdf.token
            %10 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
            %11 = ktdf.create_token : !ktdf.token
            %12 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
            %13 = ktdf.create_token : !ktdf.token
            %14 = ktdf.create_token : !ktdf.token
            %alloc = memref.alloc() : memref<1x64xf16, "L1">
            ktdf.private_yield %8, %9, %10, %11, %12, %13, %14, %alloc : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token, !ktdf.token, memref<1x64xf16, "L1">
          }
          ktdf.stage depends_in(none) depends_out(%7#5) {
            ktdf.data_transfer from %reinterpret_cast[%arg1, %arg2] size [1, 64] to %7#7[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
          }
          ktdf.stage depends_in(%7#5) depends_out(%7#3) {
            ktdf.data_transfer from %7#7[0, 0] size [1, 64] to %7#2 size [1, 64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
          }
          ktdf.stage depends_in(%7#3) depends_out(%7#1) {
            %8 = ktdf.read_from_fifo %7#2 : <"L1" -> "SFU", 64xf16> -> tensor<1x64xf16>
            %9 = tensor.empty() : tensor<1x64xf16>
            %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<1x64xf16>) outs(%9 : tensor<1x64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
            ^bb0(%in: f16, %out: f16):
              %11 = math.sqrt %in : f16
              linalg.yield %11 : f16
            } -> tensor<1x64xf16>
            ktdf.write_to_fifo %10, %7#0 : tensor<1x64xf16>, <"SFU" -> "DDR", 64xf16>
          } {applicable_units = ["SFU"]}
          ktdf.stage depends_in(%7#1) depends_out(none) {
            ktdf.data_transfer from %7#0 size [1, 64] to %reinterpret_cast_1[%arg1, %arg2] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
