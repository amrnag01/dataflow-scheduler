// RUN: dataflow-scheduler-opt -ktir-map-and-tile -allow-unregistered-dialect %s | FileCheck %s

// CHECK-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-DAG: #[[$ATTR_5:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>

// CHECK-LABEL:   func.func @local_schedule_1() attributes {grid = [1 : index]} {
// CHECK-DAG:      %[[CONSTANT_0:.*]] = arith.constant dense<-1.000000e+00> : tensor<1x4x64xf16>
// CHECK-DAG:      %[[CONSTANT_1:.*]] = arith.constant dense<0xFC00> : tensor<1x4x64xf16>
// CHECK-DAG:      %[[CONSTANT_2:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-DAG:      %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK-DAG:      %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK-DAG:      %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-DAG:      %[[CONSTANT_6:.*]] = arith.constant 64 : index
// CHECK-DAG:      %[[CONSTANT_7:.*]] = arith.constant 64000 : index
// CHECK-DAG:      %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [4, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<4x64x64xf16>
// CHECK-DAG:      %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_7]], sizes: [4, 4, 64], strides: [256, 64, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<4x4x64xf16>
// CHECK-DAG:      %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_4]]} : memref<4x64x64xf16> -> !ktdp.access_tile<4x64x64xindex>
// CHECK-DAG:      %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_5]]} : memref<4x4x64xf16> -> !ktdp.access_tile<4x4x64xindex>
// CHECK:          scf.for %[[VAL_0:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_6]] step %[[CONSTANT_6]] {
// CHECK-DAG:          %[[LOAD_0:.*]] = ktdp_lowering.load %[[CONSTRUCT_ACCESS_TILE_0]]{{\[}}%[[VAL_0]], 0, %[[VAL_1]]] [1, 64, 64] [1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : !ktdp.access_tile<4x64x64xindex> -> tensor<1x64x64xf16>
// CHECK-DAG:          %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-DAG:          %[[EMPTY_1:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-DAG:          %[[GENERIC_0:.*]]:5 = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_3]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction", "parallel", "reduction"]} outs(%[[LOAD_0]], %[[EMPTY_0]], %[[EMPTY_1]], %[[CONSTANT_1]], %[[CONSTANT_0]] : tensor<1x64x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x4x64xf16>, tensor<1x4x64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
// CHECK:              ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK-DAG:            %[[INDEX_0:.*]] = linalg.index 3 : index
// CHECK-DAG:            %[[CMPI_0:.*]] = arith.cmpi eq, %[[INDEX_0]], %[[CONSTANT_5]] : index
// CHECK-DAG:            %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[CONSTANT_2]] : f16
// CHECK-DAG:            %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDF_0]], %[[VAL_4]] : f16
// CHECK-DAG:            %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[VAL_4]], %[[VAL_3]] : f16
// CHECK-DAG:            %[[VAL_7:.*]]:4 = "spyreop_next.compare_and_swap_max"(%[[VAL_2]], %[[SELECT_1]], %[[VAL_5]], %[[VAL_6]]) : (f16, f16, f16, f16) -> (f16, f16, f16, f16)
// CHECK:                linalg.yield %[[VAL_7]]#0, %[[VAL_7]]#1, %[[SELECT_0]], %[[VAL_7]]#2, %[[VAL_7]]#3 : f16, f16, f16, f16, f16
// CHECK-NEXT:         } -> (tensor<1x64x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x4x64xf16>, tensor<1x4x64xf16>)
// CHECK-NEXT:         ktdp_lowering.store %[[VAL_8:.*]]#3 into %[[CONSTRUCT_ACCESS_TILE_1]]{{\[}}%[[VAL_0]], 0, %[[VAL_1]]] [1, 4, 64] [1, 1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x4x64xf16> into !ktdp.access_tile<4x4x64xindex>
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c2048 = arith.constant 2048 : index
    %c64000 = arith.constant 64000 : index
    %cst_neg1   = arith.constant -1.0 : f16
    %cst_neg_inf = arith.constant 0xFC00 : f16
    %splat_neg_inf     = arith.constant dense<0xFC00> : tensor<4x1x64xf16>
    %splat_neg1        = arith.constant dense<-1.0>   : tensor<4x1x64xf16>
    %splat_ones        = arith.constant dense<1.0>    : tensor<4x1x64xf16>
    %splat_neg_inf_all = arith.constant dense<0xFC00> : tensor<4x4x64xf16>
    %splat_ones_all    = arith.constant dense<1.0>    : tensor<4x4x64xf16>
    %splat_neg1_all    = arith.constant dense<-1.0>   : tensor<4x4x64xf16>

    %logits_view = ktdp.construct_memory_view %c0, sizes: [4, 64, 64], strides: [4096, 64, 1] {
      coordinate_set = affine_set<(d0,d1,d2):(d0>=0,-d0+3>=0,d1>=0,-d1+63>=0,d2>=0,-d2+63>=0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<4x64x64xf16>

    %values_view = ktdp.construct_memory_view %c64000, sizes: [4, 4, 64], strides: [256, 64, 1] {
      coordinate_set = affine_set<(d0,d1,d2):(d0>=0,-d0+3>=0,d1>=0,-d1+3>=0,d2>=0,-d2+63>=0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<4x4x64xf16>

    %logits_tile = ktdp.construct_access_tile %logits_view[%c0, %c0, %c0] {
      access_tile_set   = affine_set<(d0,d1,d2):(d0>=0,-d0+3>=0,d1>=0,-d1+63>=0,d2>=0,-d2+63>=0)>,
      access_tile_order = affine_map<(d0,d1,d2)->(d0,d1,d2)>
    } : memref<4x64x64xf16> -> !ktdp.access_tile<4x64x64xindex>
    %value_tensor = ktdp.load %logits_tile : !ktdp.access_tile<4x64x64xindex> -> tensor<4x64x64xf16>

    %index_tensor = tensor.empty() : tensor<4x64xf16>
    %counter_tensor = tensor.empty() : tensor<4x64xf16>
    // Local memory for saving the running top-k values; initialise all slots to -inf in one shot
    %top_k_val_tensor = arith.constant dense<0xFC00> : tensor<4x4x64xf16>
    // Local memory for saving the running top-k indices; initialise all slots to -1 in one shot
    %top_k_idx_tensor = arith.constant dense<-1.0> : tensor<4x4x64xf16>

    %linalg_results:5 = linalg.generic {
      indexing_maps = [
        // grp, exp, lane, k
        affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>,
        affine_map<(d0,d1,d2,d3)->(d0,d2)>,
        affine_map<(d0,d1,d2,d3)->(d0,d2)>,
        affine_map<(d0,d1,d2,d3)->(d0,d3,d2)>,
        affine_map<(d0,d1,d2,d3)->(d0,d3,d2)>
      ],
      iterator_types = ["parallel", "reduction", "parallel", "reduction"]
    }
      outs(%value_tensor, %index_tensor, %counter_tensor, %top_k_val_tensor, %top_k_idx_tensor :
            tensor<4x64x64xf16>, tensor<4x64xf16>, tensor<4x64xf16>, tensor<4x4x64xf16>, tensor<4x4x64xf16>) {
    ^bb0(%curr_val: f16, %curr_idx: f16, %counter: f16, %top_k_val: f16, %top_k_idx: f16):
      %cst_one_f16 = arith.constant 1.0 : f16

      %index = linalg.index 3 : index // Get the current loop index (k dim)
      %is_first = arith.cmpi eq, %index, %c0 : index // Check if it's k==0
      %counter_inc = arith.addf %counter, %cst_one_f16 : f16
      %counter_next = arith.select %is_first, %counter_inc, %counter : f16
      %curr_idx_selected = arith.select %is_first, %counter, %curr_idx : f16

      %res:4 = "spyreop_next.compare_and_swap_max" (%curr_val, %curr_idx_selected, %top_k_val, %top_k_idx) : (f16, f16, f16, f16) -> (f16, f16, f16, f16)

      linalg.yield %res#0, %res#1, %counter_next, %res#2, %res#3
        : f16, f16, f16, f16, f16
    } -> (tensor<4x64x64xf16>, tensor<4x64xf16>, tensor<4x64xf16>, tensor<4x4x64xf16>, tensor<4x4x64xf16>)

    // Store updated top-k values to global values_view.
    %values_tile = ktdp.construct_access_tile %values_view[%c0, %c0, %c0] {
      access_tile_set   = affine_set<(d0,d1,d2):(d0>=0,-d0+3>=0,d1>=0,-d1+3>=0,d2>=0,-d2+63>=0)>,
      access_tile_order = affine_map<(d0,d1,d2)->(d0,d1,d2)>
    } : memref<4x4x64xf16> -> !ktdp.access_tile<4x4x64xindex>
    ktdp.store %linalg_results#3, %values_tile : tensor<4x4x64xf16>, !ktdp.access_tile<4x4x64xindex>

    return
  }
}