// RUN: dataflow-scheduler-opt --custom-linalg-bufferization %s | FileCheck %s

// Verify that CustomLinalgBufferizationPass rebuilds the linalg.generic
// reading a bufferization.to_tensor so it runs on buffers, and drops the tensor
// view.
//
// The shape here is what the post_scheduling device pattern leaves behind: the
// splat operand already has a register-file buffer, while the sibling operand is
// still a tensor-typed ktdf.read_from_fifo and the result is still a
// tensor.empty() feeding ktdf.write_to_fifo.  All of them have to become buffers
// together, because linalg rejects an op mixing tensor and buffer semantics.
//
// The indexing maps must come out untouched: the broadcast map already reads the
// element the splat made uniform.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast = affine_map<(d0, d1, d2) -> (0, d1, 0)>

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[BCAST:.*]] = affine_map<(d0, d1, d2) -> (0, d1, 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

module {
  // CHECK-LABEL: func.func @absorb(
  // CHECK-SAME:    %[[BUF:[^:,]*]]: memref<1x1x32xf32, "SFU_REG">)
  func.func @absorb(%buffer: memref<1x1x32xf32, "SFU_REG">) {
    %out = tensor.empty() : tensor<1x1x32xf32>
    ktdf.pipeline {
      %slots:4 = ktdf.private -> (
          !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
          !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>,
          !ktdf.token,
          !ktdf.token) {
        %f0 = ktdf.fifo.allocate() ->
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        %f1 = ktdf.fifo.allocate() ->
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1 :
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>,
            !ktdf.token, !ktdf.token
      }
      // The compute stage names the unit, which is what says where the results
      // live: the memory local to that unit.
      ktdf.stage depends_in(%slots#2) depends_out(%slots#3) {
        // The sibling input is re-read as a memref, and the result gets a
        // buffer in the compute unit's local memory.
        // CHECK:       %[[MREAD:.*]] = ktdf.read_from_fifo %{{.*}}#0
        // CHECK-SAME:    -> memref<1x1x32xf32>
        // CHECK:       %[[OUT:.*]] = memref.alloc() : memref<1x1x32xf32, "SFU_REG">
        %r0 = ktdf.read_from_fifo %slots#0
          : <"L1LU" -> "SFU", 32xf32> -> tensor<1x1x32xf32>

        %as_tensor = bufferization.to_tensor %buffer
          : memref<1x1x32xf32, "SFU_REG"> to tensor<1x1x32xf32>

        // Buffer-mode generic: no results, same maps, splat operand is the
        // buffer the pattern materialized.
        // CHECK:       linalg.generic
        // CHECK-SAME:    indexing_maps = [#[[MAP]], #[[BCAST]], #[[MAP]]]
        // CHECK-SAME:    ins(%[[MREAD]], %[[BUF]] : memref<1x1x32xf32>, memref<1x1x32xf32, "SFU_REG">)
        // CHECK-SAME:    outs(%[[OUT]] : memref<1x1x32xf32, "SFU_REG">)
        %res = linalg.generic {
            indexing_maps = [#map, #map_bcast, #map],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%r0, %as_tensor : tensor<1x1x32xf32>, tensor<1x1x32xf32>)
            outs(%out : tensor<1x1x32xf32>) {
          ^bb0(%in: f32, %in_b: f32, %o: f32):
            %add = arith.addf %in, %in_b : f32
            linalg.yield %add : f32
        } -> tensor<1x1x32xf32>

        // The write now sends the output buffer.
        // CHECK:       ktdf.write_to_fifo %[[OUT]], %{{.*}}#1
        ktdf.write_to_fifo %res, %slots#1
          : tensor<1x1x32xf32>, <"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
    // Nothing tensor-shaped is left over.
    // CHECK-NOT:   bufferization.to_tensor
    // CHECK-NOT:   tensor.empty
    return
  }

  // The device pattern has already re-read the splat operand from slot #1 as a
  // memref and copied it into a register, after the still-tensor read of slot
  // #0.  Both slots run between the same two units, so they share one FIFO
  // that delivers in write order: the re-read of #0 must keep its place ahead
  // of #1 rather than move down to the generic.
  // CHECK-LABEL: func.func @keeps_read_order(
  func.func @keeps_read_order() {
    %out = tensor.empty() : tensor<1x1x32xf32>
    ktdf.pipeline {
      %slots:5 = ktdf.private -> (
          !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
          !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
          !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>,
          !ktdf.token,
          !ktdf.token) {
        %f0:2 = ktdf.fifo.allocate() ->
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        %f1 = ktdf.fifo.allocate() ->
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0#0, %f0#1, %f1, %t0, %t1 :
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>,
            !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(%slots#3) depends_out(%slots#4) {
        // CHECK:       %[[R0:.*]] = ktdf.read_from_fifo %{{.*}}#0
        // CHECK-SAME:    -> memref<1x1x32xf32>
        // CHECK-NEXT:  %[[R1:.*]] = ktdf.read_from_fifo %{{.*}}#1 {splat
        // CHECK-NEXT:  %[[REG:.*]] = memref.alloc() : memref<1x1x32xf32, "SFU_REG">
        // CHECK-NEXT:  memref.copy %[[R1]], %[[REG]]
        // CHECK:       linalg.generic
        // CHECK-SAME:    ins(%[[R0]], %[[REG]] :
        %r0 = ktdf.read_from_fifo %slots#0
          : <"L1LU" -> "SFU", 32xf32> -> tensor<1x1x32xf32>
        %r1 = ktdf.read_from_fifo %slots#1
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> memref<1x1x32xf32>
        %reg = memref.alloc() : memref<1x1x32xf32, "SFU_REG">
        memref.copy %r1, %reg
          : memref<1x1x32xf32> to memref<1x1x32xf32, "SFU_REG">
        %as_tensor = bufferization.to_tensor %reg
          : memref<1x1x32xf32, "SFU_REG"> to tensor<1x1x32xf32>
        %res = linalg.generic {
            indexing_maps = [#map, #map_bcast, #map],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%r0, %as_tensor : tensor<1x1x32xf32>, tensor<1x1x32xf32>)
            outs(%out : tensor<1x1x32xf32>) {
          ^bb0(%in: f32, %in_b: f32, %o: f32):
            %add = arith.addf %in, %in_b : f32
            linalg.yield %add : f32
        } -> tensor<1x1x32xf32>
        ktdf.write_to_fifo %res, %slots#2
          : tensor<1x1x32xf32>, <"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
    return
  }
}
