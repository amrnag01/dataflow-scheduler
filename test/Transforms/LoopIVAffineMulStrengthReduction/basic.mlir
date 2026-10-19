// RUN: dataflow-scheduler-opt --loop-iv-affine-mul-strength-reduction --allow-unregistered-dialect %s | FileCheck %s

// -----------------------------------------------------------------------------
// Test 1: 2D case — (d0, d1) -> (d0 * 2 + d1)
//
// The affine.apply is removed. The outer loop gains an iter_arg initialized to
// 0, the inner loop gains an iter_arg initialized from the outer iter_arg.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_2d
// CHECK-NOT:   affine.apply
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[IA_OUTER:.*]] =
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[IA_INNER:.*]] = %[[IA_OUTER]])
// CHECK:           "test.use"(%[[IA_INNER]])

func.func @test_2d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  scf.for %iv0 = %c0 to %c4 step %c1 {
    scf.for %iv1 = %c0 to %c2 step %c1 {
      %idx = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%iv0, %iv1)
      "test.use"(%idx) : (index) -> ()
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 2: 3D case — (d0, d1, d2) -> (d0 * 6 + d1 * 2 + d2)
//
// Three nested loops; three iter_args are chained.
// The affine.apply is removed; the innermost iter_arg replaces it.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_3d
// CHECK-NOT:   affine.apply
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             "test.use"(

func.func @test_3d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  scf.for %iv0 = %c0 to %c4 step %c1 {
    scf.for %iv1 = %c0 to %c3 step %c1 {
      scf.for %iv2 = %c0 to %c2 step %c1 {
        %idx = affine.apply affine_map<(d0, d1, d2) -> (d0 * 6 + d1 * 2 + d2)>(%iv0, %iv1, %iv2)
        "test.use"(%idx) : (index) -> ()
      }
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 3: middle loop does not participate — iter_arg threads through it.
//
// Outer provides d0 (coeff 8), middle is not in the map, inner provides d1
// (coeff 1).  The middle loop must receive a pass-through iter_arg.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_middle_passthrough
// CHECK-NOT:   affine.apply
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             "test.use"(

func.func @test_middle_passthrough() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  scf.for %iv0 = %c0 to %c4 step %c1 {
    scf.for %iv_mid = %c0 to %c3 step %c1 {
      scf.for %iv1 = %c0 to %c8 step %c1 {
        // iv_mid is not used in the map
        %idx = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%iv0, %iv1)
        "test.use"(%idx) : (index) -> ()
      }
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 4: loop bounds start at non-zero lower bound — affine.apply removed.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_nonzero_lb
// CHECK-NOT:   affine.apply
// CHECK:       "test.use"(

func.func @test_nonzero_lb() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  scf.for %iv0 = %c2 to %c10 step %c3 {
    scf.for %iv1 = %c1 to %c5 step %c1 {
      %idx = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%iv0, %iv1)
      "test.use"(%idx) : (index) -> ()
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 5: NEGATIVE — operand is not a loop IV (defined outside loops).
// The pass must leave the affine.apply untouched.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_no_transform_non_iv
// CHECK:       affine.apply

func.func @test_no_transform_non_iv(%arg0 : index, %arg1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv0 = %c0 to %c4 step %c1 {
    // %arg1 is a function argument, not a loop IV
    %idx = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%iv0, %arg1)
    "test.use"(%idx) : (index) -> ()
  }
  return
}

// -----------------------------------------------------------------------------
// Test 6: NEGATIVE — only one participating loop (requires >= 2).
// Both operands are the same IV — only one distinct loop.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_no_transform_single_loop
// CHECK:       affine.apply

func.func @test_no_transform_single_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv0 = %c0 to %c4 step %c1 {
    %idx = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%iv0, %iv0)
    "test.use"(%idx) : (index) -> ()
  }
  return
}

// -----------------------------------------------------------------------------
// Test 7: multiple affine.apply ops in the same loop nest are all transformed.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_multiple_applies
// CHECK-NOT:   affine.apply
// CHECK:       "test.use"(
// CHECK:       "test.use"(

func.func @test_multiple_applies() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  scf.for %iv0 = %c0 to %c4 step %c1 {
    scf.for %iv1 = %c0 to %c2 step %c1 {
      %idx0 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%iv0, %iv1)
      %idx1 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%iv0, %iv1)
      "test.use"(%idx0) : (index) -> ()
      "test.use"(%idx1) : (index) -> ()
    }
  }
  return
}

// =============================================================================
// arith.muli IV x loop-invariant factor tests
// =============================================================================

// -----------------------------------------------------------------------------
// Test 8: arith.muli with a compile-time constant factor.
// The loop gains an iter_arg; the arith.muli is removed.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_muli_constant_factor
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[IA:.*]] =
// CHECK-NOT:   arith.muli
// CHECK:         "test.use"(%[[IA]])

func.func @test_muli_constant_factor() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c12 = arith.constant 12 : index
  scf.for %iv = %c0 to %c12 step %c1 {
    %m = arith.muli %iv, %c3 : index
    "test.use"(%m) : (index) -> ()
  }
  return
}

// -----------------------------------------------------------------------------
// Test 9: arith.muli with a runtime (non-constant) loop-invariant factor.
// %factor is a function argument — still strength-reduced.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_muli_runtime_factor
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[IA:.*]] =
// CHECK-NOT:   arith.muli
// CHECK:         "test.use"(%[[IA]])

func.func @test_muli_runtime_factor(%factor : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %m = arith.muli %iv, %factor : index
    "test.use"(%m) : (index) -> ()
  }
  return
}

// -----------------------------------------------------------------------------
// Test 10: multiple arith.muli ops with the same factor are coalesced into
// one iter-arg. Both uses refer to the same iter-arg value.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_muli_same_factor_coalesced
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[IA:.*]] =
// CHECK-NOT:   arith.muli
// CHECK:         "test.use"(%[[IA]])
// CHECK:         "test.use"(%[[IA]])

func.func @test_muli_same_factor_coalesced() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %m0 = arith.muli %iv, %c4 : index
    %m1 = arith.muli %iv, %c4 : index
    "test.use"(%m0) : (index) -> ()
    "test.use"(%m1) : (index) -> ()
  }
  return
}

// -----------------------------------------------------------------------------
// Test 11: multiple arith.muli ops with different factors each get their own
// iter-arg. The loop must have two iter_args.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_muli_different_factors
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})

func.func @test_muli_different_factors() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c12 = arith.constant 12 : index
  scf.for %iv = %c0 to %c12 step %c1 {
    %m0 = arith.muli %iv, %c3 : index
    %m1 = arith.muli %iv, %c5 : index
    "test.use"(%m0) : (index) -> ()
    "test.use"(%m1) : (index) -> ()
  }
  return
}

// -----------------------------------------------------------------------------
// Test 12: NEGATIVE — arith.muli where the factor is NOT loop-invariant.
// %factor is defined inside the loop; the pass must leave it untouched.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_muli_non_invariant_factor
// CHECK:       arith.muli

func.func @test_muli_non_invariant_factor() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %factor = "test.compute"(%iv) : (index) -> index
    %m = arith.muli %iv, %factor : index
    "test.use"(%m) : (index) -> ()
  }
  return
}
