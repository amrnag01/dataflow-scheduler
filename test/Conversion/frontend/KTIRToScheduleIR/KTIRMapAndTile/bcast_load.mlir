// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// Tests that broadcast inputs are tiled into loads of stride 0. Together with
// KTIRPipeline/bcast_load.mlir, whose input is the output of this test, this
// covers broadcasts from tiling down to pipeline construction.

#id = affine_map<(d0) -> (d0)>
#bcast = affine_map<(d0) -> (0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 == 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

// The broadcast of a loaded element is folded into a load of stride 0.
// CHECK-LABEL: func.func @broadcast
// CHECK:         scf.for {{.+}} {
// CHECK:           %[[BCAST:.+]] = ktdp_lowering.load %{{.+}}[0] [64] [0]
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%{{.+}}, %[[BCAST]] :
// CHECK:         }
// CHECK-NOT:     tensor.extract_slice
func.func @broadcast(%addr_a: index, %addr_b: index, %addr_out: index) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view_a = ktdp.construct_memory_view %addr_a, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile_a = ktdp.construct_access_tile %view_a[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %view_b = ktdp.construct_memory_view %addr_b, sizes: [1], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1xf16>
  %tile_b = ktdp.construct_access_tile %view_b[%c0] {access_tile_order = #id, access_tile_set = #set1} : memref<1xf16> -> !ktdp.access_tile<1xindex>
  %view_out = ktdp.construct_memory_view %addr_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile_out = ktdp.construct_access_tile %view_out[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %a = ktdp.load %tile_a : <128xindex> -> tensor<128xf16>
  %b = ktdp.load %tile_b : <1xindex> -> tensor<1xf16>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #bcast, #id], iterator_types = ["parallel"]} ins(%a, %b : tensor<128xf16>, tensor<1xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %s = arith.addf %x, %y : f16
    linalg.yield %s : f16
  } -> tensor<128xf16>
  ktdp.store %result, %tile_out : tensor<128xf16>, <128xindex>
  return
}

// The broadcast of a loaded element is folded into a load of stride 0 before
// its via.
// CHECK-LABEL: func.func @broadcast_through_via
// CHECK:         scf.for {{.+}} {
// CHECK:           %[[BCAST:.+]] = ktdp_lowering.load %{{.+}}[0] [64] [0]
// CHECK:           %[[HOP:.+]] = ktdf.via["L1"] %[[BCAST]]
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%{{.+}}, %[[HOP]] :
// CHECK:         }
// CHECK-NOT:     tensor.extract_slice
func.func @broadcast_through_via(%addr_a: index, %addr_b: index, %addr_out: index) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view_a = ktdp.construct_memory_view %addr_a, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile_a = ktdp.construct_access_tile %view_a[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %view_b = ktdp.construct_memory_view %addr_b, sizes: [1], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1xf16>
  %tile_b = ktdp.construct_access_tile %view_b[%c0] {access_tile_order = #id, access_tile_set = #set1} : memref<1xf16> -> !ktdp.access_tile<1xindex>
  %view_out = ktdp.construct_memory_view %addr_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile_out = ktdp.construct_access_tile %view_out[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %a = ktdp.load %tile_a : <128xindex> -> tensor<128xf16>
  %b = ktdp.load %tile_b : <1xindex> -> tensor<1xf16>
  %b_l1 = ktdf.via["L1"] %b : tensor<1xf16>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #bcast, #id], iterator_types = ["parallel"]} ins(%a, %b_l1 : tensor<128xf16>, tensor<1xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %s = arith.addf %x, %y : f16
    linalg.yield %s : f16
  } -> tensor<128xf16>
  ktdp.store %result, %tile_out : tensor<128xf16>, <128xindex>
  return
}
