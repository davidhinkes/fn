# Layer API Refactoring - Output Parameters

## Summary

The Layer interface has been refactored to use output parameters instead of return values. This change is designed to reduce memory allocations by allowing callers to reuse pre-allocated buffers.

## Changes Made

### Layer Interface (layer.go)

**Before:**
```go
type Layer interface {
    F(x mat.Vector, h []float64) mat.Vector
    D(x mat.Vector, h []float64) (dYdX mat.Matrix, dYdH mat.Matrix)
    NumWeights() int
}
```

**After:**
```go
type Layer interface {
    F(dst *mat.VecDense, x mat.Vector, h []float64)
    D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64)
    NumWeights() int
}
```

### Key Changes

1. **F method**: Now takes a `*mat.VecDense` destination parameter that is filled with the output
2. **D method**: Now takes `*mat.Dense` destination parameters for both dYdX and dYdH derivatives
3. **Nil handling**: Passing `nil` for `dYdH` indicates that weight derivatives should not be computed (useful for layers with no weights)

## Implementation Status

All core components have been updated:

- ✅ Layer interface (layer.go)
- ✅ Serial composition (serial.go)
- ✅ Parallel composition (parallel.go)
- ✅ Model (model.go)
- ✅ Training (train.go)
- ✅ All layer implementations:
  - Perceptron
  - Bias
  - Sigmoid (via StaticFunc)
  - ReLU (via StaticFunc)
  - Radial
  - Scalar
- ✅ All tests updated and passing
- ✅ Example programs compile and run

## Memory Allocation Benefits

### Before (Return Values)
Each call to `F()` and `D()` allocated new matrices/vectors:
- `F()`: Allocated new VecDense for output
- `D()`: Allocated 2 new Dense matrices (dYdX and dYdH)
- In serial composition: Multiple intermediate allocations per forward/backward pass
- In training loop: Allocations multiplied by batch size and iterations

### After (Output Parameters)
Callers can now:
- Pre-allocate destination buffers once
- Reuse buffers across multiple calls
- Use `ReuseAs()` methods to resize buffers without reallocation when dimensions match
- Skip computing unused derivatives by passing nil

### Example Optimization Opportunity

In training loops, the same gradient computation happens repeatedly:
```go
// Before: Allocates new matrices each iteration
for i := 0; i < iterations; i++ {
    dYdX, dYdW := layer.D(x, weights)  // 2 allocations
    // use derivatives...
}

// After: Reuse buffers
var dYdX, dYdW mat.Dense
for i := 0; i < iterations; i++ {
    layer.D(&dYdX, &dYdW, x, weights)  // Reuses existing buffers
    // use derivatives...
}
```

## Testing

All existing tests pass:
```
=== RUN   TestBiasLayer
--- PASS: TestBiasLayer (0.01s)
=== RUN   TestPerceptronLayer
--- PASS: TestPerceptronLayer (0.05s)
=== RUN   TestScalarLayer
--- PASS: TestScalarLayer (3.79s)
=== RUN   TestStaticFuncLayer
--- PASS: TestStaticFuncLayer (0.58s)
=== RUN   TestParallel
--- PASS: TestParallel (0.00s)
[... all tests passing ...]
```

Example programs run successfully:
- binary_integer_example: ✅ Compiles and trains
- regression_example: ✅ Compiles and trains

## Next Steps (If Proceeding with Full Adoption)

1. **Benchmark**: Create detailed benchmarks comparing allocation counts before/after
2. **Pool optimization**: Consider using sync.Pool for frequently allocated matrices in hot paths
3. **API refinement**: Evaluate if any further optimizations are possible
4. **Documentation**: Update godoc comments with performance guidance
5. **Migration guide**: Document for external users if this becomes public

## Notes

- The refactoring maintains backward compatibility in behavior (same computation results)
- All layers properly handle the nil dYdH parameter for weight-free layers
- The serial and parallel composition layers correctly chain the new API
- ReuseAs() is used appropriately to resize buffers when needed
