// RUN: dataflow-scheduler-opt --ktir-bufferize %s | FileCheck %s

// Tests that stores of loaded data are converted to data transfers between
// memories.

#id = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

// A store of loaded data becomes a single data transfer between the memories.
// CHECK-LABEL: func.func @store_of_load
// CHECK-NOT:     ktdp_lowering
// CHECK:         ktdf.data_transfer from %{{.+}}[0] size [64] to %{{.+}}[0] size [64]
// CHECK-NOT:     ktdp_lowering
func.func @store_of_load(%addr_in: index, %addr_out: index) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view_in = ktdp.construct_memory_view %addr_in, sizes: [64], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
  %tile_in = ktdp.construct_access_tile %view_in[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %view_out = ktdp.construct_memory_view %addr_out, sizes: [64], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
  %tile_out = ktdp.construct_access_tile %view_out[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %data = ktdp.load %tile_in : <64xindex> -> tensor<64xf16>
  ktdp.store %data, %tile_out : tensor<64xf16>, <64xindex>
  return
}

// Reshapes between the load and the store are folded into the data transfer.
// CHECK-LABEL: func.func @store_of_reshaped_load
// CHECK-NOT:     tensor.expand_shape
// CHECK:         ktdf.data_transfer from %{{.+}}[0] size [64] to %{{.+}}[0, 0] size [1, 64]
// CHECK-NOT:     tensor.expand_shape
func.func @store_of_reshaped_load(%addr_in: index, %addr_out: index) attributes {grid = [1 : index]} {
  %c0 = arith.constant 0 : index
  %view_in = ktdp.construct_memory_view %addr_in, sizes: [64], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
  %tile_in = ktdp.construct_access_tile %view_in[%c0] {access_tile_order = #id, access_tile_set = #set} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %view_out = ktdp.construct_memory_view %addr_out, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
  %tile_out = ktdp.construct_access_tile %view_out[%c0, %c0] {access_tile_order = #id2, access_tile_set = #set2} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %data = ktdp.load %tile_in : <64xindex> -> tensor<64xf16>
  %expanded = tensor.expand_shape %data [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
  ktdp.store %expanded, %tile_out : tensor<1x64xf16>, <1x64xindex>
  return
}
