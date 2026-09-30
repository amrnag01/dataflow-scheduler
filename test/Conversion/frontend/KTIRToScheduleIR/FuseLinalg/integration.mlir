// RUN: dataflow-scheduler-opt --convert-elementwise-to-linalg --linalg-morph-ops="category-to-generic=true named-to-generic=true" --fuse-linalg %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:    %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[CONSTANT_1:.*]] = arith.constant 3 : index
// CHECK-NEXT:    %[[CONSTANT_2:.*]] = arith.constant 1024 : index
// CHECK-NEXT:    %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:    %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:    %[[CONSTANT_5:.*]] = arith.constant 24576 : index
// CHECK-NEXT:    %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:    %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:    %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:    %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:    %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:    %[[CONSTRUCT_MEMORY_VIEW_3:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:    %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG0]] : index
// CHECK-NEXT:    %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[ADDI_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:    %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[ADDI_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:    %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[ADDI_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:    %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <1x64xindex> -> tensor<1x64xf16>
// CHECK-NEXT:    %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <1x64xindex> -> tensor<1x64xf16>
// CHECK-NEXT:    %[[LOAD_2:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_2]] : <1x64xindex> -> tensor<1x64xf16>
// CHECK-NEXT:    %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:    %[[EMPTY_1:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:    %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_2]] : tensor<1x64xf16>) outs(%[[LOAD_2]] : tensor<1x64xf16>) attrs =  {ktdf_arch.maps_to = "SFU"} {
// CHECK-NEXT:    ^bb0(%[[VAL_0:.*]]: f16, %[[VAL_1:.*]]: f16):
// CHECK-NEXT:      %[[SQRT_0:.*]] = math.sqrt %[[VAL_0]] {ktdf_arch.maps_to = "SFU"} : f16
// CHECK-NEXT:      linalg.yield %[[SQRT_0]] : f16
// CHECK-NEXT:    } -> tensor<1x64xf16>
// CHECK-NEXT:    %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_0]], %[[LOAD_0]], %[[LOAD_1]] : tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:    ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:      %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f16
// CHECK-NEXT:      %[[ADDF_1:.*]] = arith.addf %[[VAL_2]], %[[ADDF_0]] : f16
// CHECK-NEXT:      linalg.yield %[[ADDF_1]] : f16
// CHECK-NEXT:    } -> tensor<1x64xf16>
// CHECK-NEXT:    %[[CONSTRUCT_ACCESS_TILE_3:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_3]]{{\[}}%[[ADDI_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:    ktdp.store %[[GENERIC_1]], %[[CONSTRUCT_ACCESS_TILE_3]] : tensor<1x64xf16>, <1x64xindex>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

module {
    ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %tile_size = arith.constant 3 : index
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %D_start_address = arith.constant 18432 : index
        %E_start_address = arith.constant 24576 : index

        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index

        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.memory_space<global>
        } : memref<96x64xf16>

        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.memory_space<global>
        } : memref<96x64xf16>

        // Construct a memory view of D from a given address
        %D_view = ktdp.construct_memory_view %D_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.memory_space<global>
        } : memref<96x64xf16>

        // Construct a memory view of E from a given address
        %E_view = ktdp.construct_memory_view %E_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.memory_space<global>
        } : memref<96x64xf16>

        %start_row_i = arith.addi %start_row, %i : index

        // Construct an access tile from the memory view of A
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Construct an access tile from the memory view of B
        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Construct an access tile from the memory view of B
        %D_access_tile = ktdp.construct_access_tile %D_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
        %D_data_tile = ktdp.load %D_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %E_data_tile = tensor.empty() : tensor<1x64xf16>

        // Perform add operation on the data tiles.
        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        %result = linalg.add
            ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
            outs(%C_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        %D_sqrt = math.sqrt %D_data_tile {ktdf_arch.maps_to = "SFU"} : tensor<1x64xf16>

        %result_2 = linalg.add {ktdf_arch.maps_to = "SFU"}
            ins(%D_sqrt, %result : tensor<1x64xf16>, tensor<1x64xf16>)
            outs(%E_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        // Construct an access tile from the memory view of E
        %E_access_tile = ktdp.construct_access_tile %E_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %result_2, %E_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

        return
    }
}
