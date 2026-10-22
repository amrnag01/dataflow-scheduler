// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt | FileCheck %s

// Round-trip tests for ktdp_lowering.ind_load and ktdp_lowering.ind_store.

// -----------------------------------------------------------------------
// Test 1: ind_load — gather, rank reduction [1,1,64] -> tensor<64xf16>
// -----------------------------------------------------------------------

// CHECK-LABEL: func @ind_load_rank_reduce(
// CHECK-SAME:    [[IAB:%arg[0-9]+]]: memref<32xi32, strided<[1], offset: ?>, "IAB">
// CHECK-SAME:    [[IDX:%arg[0-9]+]]: index
// CHECK-SAME:    [[BASE:%arg[0-9]+]]: memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
// CHECK-SAME:    [[O1:%arg[0-9]+]]: index
// CHECK:        ktdp_lowering.ind_load [[IAB]][[[IDX]]] [[BASE]][0, [[O1]], 0] [1, 1, 64] [1, 1, 1]
// CHECK-SAME:   : memref<32xi32, strided<[1], offset: ?>, "IAB">,
// CHECK-SAME:     memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global"> -> tensor<64xf16>
func.func @ind_load_rank_reduce(
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">,
    %o1   : index) -> tensor<64xf16> {
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, %o1, 0] [1, 1, 64] [1, 1, 1]
      : memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<64xf16>
  return %v : tensor<64xf16>
}

// -----------------------------------------------------------------------
// Test 2: ind_load — no rank reduction [1,2,64] -> tensor<1x2x64xf16>
// -----------------------------------------------------------------------

// CHECK-LABEL: func @ind_load_no_reduce(
// CHECK:        ktdp_lowering.ind_load {{.*}} [1, 2, 64] [1, 1, 1]
// CHECK-SAME:   -> tensor<1x2x64xf16>
func.func @ind_load_no_reduce(
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">)
    -> tensor<1x2x64xf16> {
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, 0, 0] [1, 2, 64] [1, 1, 1]
      : memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<1x2x64xf16>
  return %v : tensor<1x2x64xf16>
}

// -----------------------------------------------------------------------
// Test 3: ind_store — scatter, rank reduction [1,2,64] -> tensor<2x64xf16>
// -----------------------------------------------------------------------

// CHECK-LABEL: func @ind_store_rank_reduce(
// CHECK-SAME:    [[SRC:%arg[0-9]+]]: tensor<2x64xf16>
// CHECK-SAME:    [[IAB:%arg[0-9]+]]: memref<32xi32, strided<[1], offset: ?>, "IAB">
// CHECK-SAME:    [[IDX:%arg[0-9]+]]: index
// CHECK-SAME:    [[BASE:%arg[0-9]+]]: memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
// CHECK:        ktdp_lowering.ind_store [[SRC]] into [[IAB]][[[IDX]]] [[BASE]][0, 0, 0] [1, 2, 64] [1, 1, 1]
// CHECK-SAME:   : tensor<2x64xf16>,
// CHECK-SAME:     memref<32xi32, strided<[1], offset: ?>, "IAB">,
// CHECK-SAME:     memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
func.func @ind_store_rank_reduce(
    %src  : tensor<2x64xf16>,
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">) {
  ktdp_lowering.ind_store %src into %iab[%idx] %base[0, 0, 0] [1, 2, 64] [1, 1, 1]
      : tensor<2x64xf16>,
        memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
  return
}

// -----------------------------------------------------------------------
// Test 4: ind_store — no rank reduction [1,2,64] -> tensor<1x2x64xf16>
// -----------------------------------------------------------------------

// CHECK-LABEL: func @ind_store_no_reduce(
// CHECK:        ktdp_lowering.ind_store {{.*}} [1, 2, 64] [1, 1, 1]
// CHECK-SAME:   : tensor<1x2x64xf16>,
func.func @ind_store_no_reduce(
    %src  : tensor<1x2x64xf16>,
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">) {
  ktdp_lowering.ind_store %src into %iab[%idx] %base[0, 0, 0] [1, 2, 64] [1, 1, 1]
      : tensor<1x2x64xf16>,
        memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
  return
}

// -----------------------------------------------------------------------
// Test 5: ind_load — index-typed IAB elements
// -----------------------------------------------------------------------

// CHECK-LABEL: func @ind_load_index_iab(
// CHECK:        ktdp_lowering.ind_load {{.*}} -> tensor<64xf16>
func.func @ind_load_index_iab(
    %iab  : memref<32xindex, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">)
    -> tensor<64xf16> {
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : memref<32xindex, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<64xf16>
  return %v : tensor<64xf16>
}
