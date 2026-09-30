// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG0]] : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_10:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_11:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:     %[[CONSTANT_12:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_12]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<96x64xf16> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[CONSTANT_13:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[MULI_2:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_13]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<96x64xf16> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[MULI_2]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[CONSTANT_14:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[MULI_3:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_14]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<96x64xf16> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: {{\[}}%[[MULI_3]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_9]] to %[[CONSTANT_10]] step %[[CONSTANT_11]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:6 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:             %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[FIFO_0]]#0, %[[FIFO_0]]#1, %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]]#3) {
// CHECK-NEXT:             ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] to %[[PRIVATE_0]]#0 size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:             ktdf.data_transfer from %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] to %[[PRIVATE_0]]#1 size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0]]#3) depends_out(%[[PRIVATE_0]]#4) {
// CHECK-NEXT:             %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#0 : <"DDR" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:             %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#1 : <"DDR" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:             %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]] : tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) attrs =  {ktdf_arch.maps_to = "SFU"} {
// CHECK-NEXT:             ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK-NEXT:               %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK-NEXT:               linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:             } -> tensor<1x64xf16>
// CHECK-NEXT:             ktdf.write_to_fifo %[[GENERIC_0]], %[[PRIVATE_0]]#2 : tensor<1x64xf16>, <"SFU" -> "DDR", 64xf16>
// CHECK-NEXT:           } {applicable_units = ["SFU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0]]#4) depends_out(%[[PRIVATE_0]]#5) {
// CHECK-NEXT:             ktdf.data_transfer from %[[PRIVATE_0]]#2 size [1, 64] to %[[REINTERPRET_CAST_2]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#id = affine_map<(d0, d1) -> (d0, d1)>

module {
    ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %tile_size = arith.constant 3 : index
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %C_start_address = arith.constant 18432 : index
    
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
    
        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
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

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        // C[start_row+i:start_row+i+2][c0:c0+63] = A[start_row+i:start_row+i+2][c0:c0+63] + B[start_row+i:start_row+i+2][c0:c0+63]
        // Perform add operation on the data tiles.
        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        %result = linalg.generic {indexing_maps=[#id, #id, #id], iterator_types=["parallel", "parallel"]}
            ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
            outs(%C_data_tile: tensor<1x64xf16>) {
        ^bb0(%a: f16, %b: f16, %c: f16):
            %sum = arith.addf %a, %b : f16
            linalg.yield %sum : f16
        } -> tensor<1x64xf16>

        // Construct an access tile from the memory view of C
        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %result, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
        
        return
    }
}
