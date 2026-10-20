// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// Tests that the order of via hops is preserved when tiling loads and stores.
// The memory spaces of the hops are synthetic.

#id = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

// The hops of a load are kept in order.
// CHECK-LABEL: func.func @load_hops
// CHECK:         scf.for {{.+}} {
// CHECK:           %[[LOAD:.+]] = ktdp_lowering.load
// CHECK:           ktdf.via["A", "B", "C", "D"] %[[LOAD]]
// CHECK:         }
func.func @load_hops(%addr: index) -> tensor<128xf16> attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile = ktdp.construct_access_tile %view[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %in = ktdp.load %tile : <128xindex> -> tensor<128xf16>
  %hop = ktdf.via["A", "B", "C", "D"] %in : tensor<128xf16>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%hop : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  return %result : tensor<128xf16>
}

// The hops of a store are kept in order.
// CHECK-LABEL: func.func @store_hops
// CHECK:         scf.for {{.+}} {
// CHECK:           %[[HOP:.+]] = ktdf.via["D", "C", "B", "A"]
// CHECK:           ktdp_lowering.store %[[HOP]]
// CHECK:         }
func.func @store_hops(%data: tensor<128xf16>, %addr: index) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile = ktdp.construct_access_tile %view[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%data : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  %hop = ktdf.via["D", "C", "B", "A"] %result : tensor<128xf16>
  ktdp.store %hop, %tile : tensor<128xf16>, <128xindex>
  return
}

// The hops of a load move after its reshape, and are kept in order.
// CHECK-LABEL: func.func @load_hops_reshaped
// CHECK:         scf.for {{.+}} {
// CHECK:           %[[LOAD:.+]] = ktdp_lowering.load
// CHECK:           %[[COLLAPSED:.+]] = tensor.collapse_shape %[[LOAD]]
// CHECK:           ktdf.via["A", "B", "C", "D"] %[[COLLAPSED]]
// CHECK:         }
func.func @load_hops_reshaped(%addr: index) -> tensor<128xf16> attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view = ktdp.construct_memory_view %addr, sizes: [1, 128], strides: [128, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<1x128xf16>
  %tile = ktdp.construct_access_tile %view[%c0, %c0] {access_tile_order = #id2, access_tile_set = #set2} : memref<1x128xf16> -> !ktdp.access_tile<1x128xindex>
  %in = ktdp.load %tile : <1x128xindex> -> tensor<1x128xf16>
  %hop = ktdf.via["A", "B", "C", "D"] %in : tensor<1x128xf16>
  %collapsed = tensor.collapse_shape %hop [[0, 1]] : tensor<1x128xf16> into tensor<128xf16>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%collapsed : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  return %result : tensor<128xf16>
}

// The hops of a store are kept in order after its reshape.
// CHECK-LABEL: func.func @store_hops_reshaped
// CHECK:         scf.for {{.+}} {
// CHECK:           %[[EXPANDED:.+]] = tensor.expand_shape
// CHECK:           %[[HOP:.+]] = ktdf.via["D", "C", "B", "A"] %[[EXPANDED]]
// CHECK:           ktdp_lowering.store %[[HOP]]
// CHECK:         }
func.func @store_hops_reshaped(%data: tensor<128xf16>, %addr: index) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view = ktdp.construct_memory_view %addr, sizes: [1, 128], strides: [128, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<1x128xf16>
  %tile = ktdp.construct_access_tile %view[%c0, %c0] {access_tile_order = #id2, access_tile_set = #set2} : memref<1x128xf16> -> !ktdp.access_tile<1x128xindex>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%data : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  %expanded = tensor.expand_shape %result [[0, 1]] output_shape [1, 128] : tensor<128xf16> into tensor<1x128xf16>
  %hop = ktdf.via["D", "C", "B", "A"] %expanded : tensor<1x128xf16>
  ktdp.store %hop, %tile : tensor<1x128xf16>, <1x128xindex>
  return
}
