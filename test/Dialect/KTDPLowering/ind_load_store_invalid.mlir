// RUN: dataflow-scheduler-opt -split-input-file -verify-diagnostics %s

// -----

// ind_load: IAB rank != 1 (rank 2)
func.func @ind_load_iab_rank2(
    %iab  : memref<4x32xi32, strided<[32, 1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">)
    -> tensor<64xf16> {
  // expected-error@+1 {{operand #0 must be 1D memref of signless integer or index values}}
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : memref<4x32xi32, strided<[32, 1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<64xf16>
  return %v : tensor<64xf16>
}

// -----

// ind_load: IAB element type is f32 (not integer or index)
func.func @ind_load_iab_f32_elem(
    %iab  : memref<32xf32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">)
    -> tensor<64xf16> {
  // expected-error@+1 {{operand #0 must be 1D memref of signless integer or index values}}
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : memref<32xf32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<64xf16>
  return %v : tensor<64xf16>
}

// -----

// ind_load: non-unit stride
func.func @ind_load_nonunit_stride(
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">)
    -> tensor<64xf16> {
  // expected-error@+1 {{attribute 'static_strides' failed to satisfy constraint: i64 dense array attribute whose elements are all 1}}
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, 0, 0] [1, 1, 64] [1, 1, 2]
      : memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<64xf16>
  return %v : tensor<64xf16>
}

// -----

// ind_load: tensor shape is not a rank reduction of sizes
// sizes = [1, 1, 64] but tensor = tensor<2x64xf16>
func.func @ind_load_bad_shape(
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">)
    -> tensor<2x64xf16> {
  // expected-error@+1 {{failed to verify that tensor shape must be a rank reduction of static_sizes}}
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<2x64xf16>
  return %v : tensor<2x64xf16>
}

// -----

// ind_load: element type mismatch (base f16, tensor i8)
func.func @ind_load_elem_mismatch(
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">)
    -> tensor<64xi8> {
  // expected-error@+1 {{failed to verify that all of {base, result} have same element type}}
  %v = ktdp_lowering.ind_load %iab[%idx] %base[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
      -> tensor<64xi8>
  return %v : tensor<64xi8>
}

// -----

// ind_store: IAB rank != 1 (rank 2)
func.func @ind_store_iab_rank2(
    %src  : tensor<2x64xf16>,
    %iab  : memref<4x32xi32, strided<[32, 1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">) {
  // expected-error@+1 {{operand #1 must be 1D memref of signless integer or index values}}
  ktdp_lowering.ind_store %src into %iab[%idx] %base[0, 0, 0] [1, 2, 64] [1, 1, 1]
      : tensor<2x64xf16>,
        memref<4x32xi32, strided<[32, 1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
  return
}

// -----

// ind_store: non-unit stride
func.func @ind_store_nonunit_stride(
    %src  : tensor<2x64xf16>,
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">) {
  // expected-error@+1 {{attribute 'static_strides' failed to satisfy constraint: i64 dense array attribute whose elements are all 1}}
  ktdp_lowering.ind_store %src into %iab[%idx] %base[0, 0, 0] [1, 2, 64] [2, 1, 1]
      : tensor<2x64xf16>,
        memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
  return
}

// -----

// ind_store: tensor shape not a rank reduction of sizes
// sizes = [1, 2, 64] but source = tensor<4x64xf16>
func.func @ind_store_bad_shape(
    %src  : tensor<4x64xf16>,
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">) {
  // expected-error@+1 {{failed to verify that tensor shape must be a rank reduction of static_sizes}}
  ktdp_lowering.ind_store %src into %iab[%idx] %base[0, 0, 0] [1, 2, 64] [1, 1, 1]
      : tensor<4x64xf16>,
        memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
  return
}

// -----

// ind_store: element type mismatch (base f16, source i8)
func.func @ind_store_elem_mismatch(
    %src  : tensor<2x64xi8>,
    %iab  : memref<32xi32, strided<[1], offset: ?>, "IAB">,
    %idx  : index,
    %base : memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">) {
  // expected-error@+1 {{failed to verify that all of {base, source} have same element type}}
  ktdp_lowering.ind_store %src into %iab[%idx] %base[0, 0, 0] [1, 2, 64] [1, 1, 1]
      : tensor<2x64xi8>,
        memref<32xi32, strided<[1], offset: ?>, "IAB">,
        memref<64x2x64xf16, strided<[8192, 64, 1], offset: ?>, "global">
  return
}
