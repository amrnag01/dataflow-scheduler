// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// Tests that insertions into loop-carried tensors are only dropped if they are
// never observed, including after the loop nest.

#id = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

// A result that is returned keeps its insertions.
// CHECK-LABEL: func.func @returned_result
// CHECK:         %[[NEST:.+]] = scf.for {{.+}} iter_args
// CHECK-SAME:      {
// CHECK:           tensor.insert_slice
// CHECK:         }
// CHECK:         return %[[NEST]]
func.func @returned_result(%addr: index) -> tensor<128xf16> attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile = ktdp.construct_access_tile %view[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %in = ktdp.load %tile : <128xindex> -> tensor<128xf16>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%in : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  return %result : tensor<128xf16>
}

// A result that is never used drops its insertions.
// CHECK-LABEL: func.func @dead_result
// CHECK-NOT:     iter_args
// CHECK:         scf.for {{.+}} {
// CHECK-NOT:       tensor.insert_slice
// CHECK:           ktdp_lowering.store
// CHECK:         }
// CHECK-NOT:     tensor.insert_slice
func.func @dead_result(%addr_in: index, %addr_out: index) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view_in = ktdp.construct_memory_view %addr_in, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile_in = ktdp.construct_access_tile %view_in[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %view_out = ktdp.construct_memory_view %addr_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile_out = ktdp.construct_access_tile %view_out[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %in = ktdp.load %tile_in : <128xindex> -> tensor<128xf16>
  %init = tensor.empty() : tensor<128xf16>
  %result:2 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%in : tensor<128xf16>) outs(%init, %init : tensor<128xf16>, tensor<128xf16>) {
  ^bb0(%x: f16, %o0: f16, %o1: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y, %y : f16, f16
  } -> (tensor<128xf16>, tensor<128xf16>)
  ktdp.store %result#0, %tile_out : tensor<128xf16>, <128xindex>
  return
}
