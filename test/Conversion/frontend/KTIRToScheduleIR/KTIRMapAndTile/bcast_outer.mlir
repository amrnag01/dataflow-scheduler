// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// The second operand is broadcast along the middle dimension, outside the
// lanes. A tile of it is already one vector, so it must not be widened.

// CHECK-LABEL: func.func @local_schedule_1()
// CHECK:         scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:        scf.for %[[IV2:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-DAG:           %[[LOAD_0:.+]] = ktdp_lowering.load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]]] [1, 1, 64] [1, 1, 1] {{.*}} -> tensor<1x1x64xf16>
// CHECK-DAG:           %[[LOAD_1:.+]] = ktdp_lowering.load %{{.+}}[%[[IV0]], 0, %[[IV2]]] [1, 1, 64] [1, 0, 1] {{.*}} -> tensor<1x1x64xf16>
// CHECK-NOT:           tensor.extract_slice
// CHECK:               linalg.generic
// CHECK-SAME:            ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
// CHECK-SAME:            outs(%{{.+}} : tensor<1x1x64xf16>)

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_bcast = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c8192 = arith.constant 8192 : index
    %c12288 = arith.constant 12288 : index
    %0 = ktdp.construct_memory_view %c4096, sizes: [4, 8, 64], strides: [512, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x8x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x8x64xf16> -> !ktdp.access_tile<4x8x64xindex>
    %2 = ktdp.construct_memory_view %c8192, sizes: [4, 8, 64], strides: [512, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x8x64xf16>
    %3 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set_bcast} : memref<4x8x64xf16> -> !ktdp.access_tile<4x1x64xindex>
    %4 = ktdp.construct_memory_view %c12288, sizes: [4, 8, 64], strides: [512, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x8x64xf16>
    %5 = ktdp.construct_access_tile %4[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x8x64xf16> -> !ktdp.access_tile<4x8x64xindex>
    %6 = ktdp.load %1 : <4x8x64xindex> -> tensor<4x8x64xf16>
    %7 = ktdp.load %3 : <4x1x64xindex> -> tensor<4x1x64xf16>
    %8 = tensor.empty() : tensor<4x8x64xf16>
    %9 = linalg.generic {indexing_maps = [#map, #map_bcast, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%6, %7 : tensor<4x8x64xf16>, tensor<4x1x64xf16>)
      outs(%8 : tensor<4x8x64xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %10 = arith.addf %in, %in_0 : f16
      linalg.yield %10 : f16
    } -> tensor<4x8x64xf16>
    ktdp.store %9, %5 : tensor<4x8x64xf16>, <4x8x64xindex>
    return
  }
}
