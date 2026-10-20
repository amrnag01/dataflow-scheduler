// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// Tests that stored data tiles with other users are tiled without changing what
// those users observe, and that stores only move into loop nests where that is
// safe.

#id = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

// A user between the loop nest and the store reads the stored data back, for
// which the access tile is computed in front of the loop nest.
// CHECK-LABEL: func.func @reload_moves_access_tile
// CHECK:         %[[TILE:.+]] = ktdp.construct_access_tile
// CHECK:         scf.for {{.+}} {
// CHECK:           ktdp_lowering.store {{.+}} into %[[TILE]]
// CHECK:         }
// CHECK:         %[[RELOAD:.+]] = ktdp.load %[[TILE]]
// CHECK:         %[[ELEM:.+]] = tensor.extract %[[RELOAD]]
// CHECK:         return %[[ELEM]]
func.func @reload_moves_access_tile(%data: tensor<128xf16>, %addr: index) -> f16 attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%data : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  %elem = tensor.extract %result[%c0] : tensor<128xf16>
  %view = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile = ktdp.construct_access_tile %view[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  ktdp.store %result, %tile : tensor<128xf16>, <128xindex>
  return %elem : f16
}

// A user of the unreshaped data reads the stored data back, reshaped back.
// CHECK-LABEL: func.func @reload_reshaped
// CHECK:         %[[TILE:.+]] = ktdp.construct_access_tile
// CHECK:         scf.for {{.+}} {
// CHECK:           ktdp_lowering.store {{.+}} into %[[TILE]]
// CHECK:         }
// CHECK:         %[[RELOAD:.+]] = ktdp.load %[[TILE]]
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[RELOAD]]
// CHECK:         return %[[COLLAPSED]]
func.func @reload_reshaped(%data: tensor<128xf16>, %addr: index) -> tensor<128xf16> attributes {grid = [1 : index]} {
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
  ktdp.store %expanded, %tile : tensor<1x128xf16>, <1x128xindex>
  return %result : tensor<128xf16>
}

// A user of the via reads the stored data back through the same via.
// CHECK-LABEL: func.func @reload_through_via
// CHECK:         %[[TILE:.+]] = ktdp.construct_access_tile
// CHECK:         scf.for {{.+}} {
// CHECK:           ktdp_lowering.store {{.+}} into %[[TILE]]
// CHECK:         }
// CHECK:         %[[RELOAD:.+]] = ktdp.load %[[TILE]]
// CHECK:         %[[HOP:.+]] = ktdf.via["A", "B"] %[[RELOAD]]
// CHECK:         return %[[HOP]]
func.func @reload_through_via(%data: tensor<128xf16>, %addr: index) -> tensor<128xf16> attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile = ktdp.construct_access_tile %view[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%data : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  %hop = ktdf.via["A", "B"] %result : tensor<128xf16>
  ktdp.store %hop, %tile : tensor<128xf16>, <128xindex>
  return %hop : tensor<128xf16>
}

// Memory effects between the loop nest and the store keep the store in place.
// CHECK-LABEL: func.func @effect_before_store
// CHECK:         %[[NEST:.+]] = scf.for {{.+}} iter_args
// CHECK-SAME:      {
// CHECK:         }
// CHECK:         memref.store
// CHECK:         ktdp.store %[[NEST]]
func.func @effect_before_store(%data: tensor<128xf16>, %addr: index, %other: memref<128xf16>) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile = ktdp.construct_access_tile %view[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%data : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  %f = arith.constant 0.0 : f16
  memref.store %f, %other[%c0] : memref<128xf16>
  ktdp.store %result, %tile : tensor<128xf16>, <128xindex>
  return
}

// An access tile that can not be computed in front of the loop nest keeps
// the store in place.
// CHECK-LABEL: func.func @immovable_access_tile
// CHECK:         %[[NEST:.+]] = scf.for {{.+}} iter_args
// CHECK-SAME:      {
// CHECK:         }
// CHECK:         %[[TILE:.+]] = ktdp.construct_access_tile
// CHECK:         ktdp.store %[[NEST]], %[[TILE]]
func.func @immovable_access_tile(%data: tensor<128xf16>, %addr: index, %offsets: memref<1xindex>) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<128xf16>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%data : tensor<128xf16>) outs(%init : tensor<128xf16>) {
  ^bb0(%x: f16, %o: f16):
    %y = spyreop.exp %x : f16
    linalg.yield %y : f16
  } -> tensor<128xf16>
  %offset = memref.load %offsets[%c0] : memref<1xindex>
  %view = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %tile = ktdp.construct_access_tile %view[%offset] {access_tile_order = #id, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
  ktdp.store %result, %tile : tensor<128xf16>, <128xindex>
  return
}
