# Memory Allocation Analysis

## Training Loop Allocation Comparison

### Example Network
```go
// 5-layer network: 32 -> 16 -> sigmoid -> 16 -> 32
model := MakeModel(
    layers.MakePerceptronLayer(32, 16),
    layers.MakeBiasLayer(16),
    layers.MakeSigmoid(),
    layers.MakePerceptronLayer(16, 32),
    layers.MakeBiasLayer(32),
)
```

## Per-Training-Step Allocations (Single Example)

### Forward Pass (F method calls)

**Old API (Return Values):**
- Perceptron(32->16): Allocates VecDense(16)
- Bias(16): Allocates VecDense(16)
- Sigmoid(16): Allocates VecDense(16)
- Perceptron(16->32): Allocates VecDense(32)
- Bias(32): Allocates VecDense(32)
- **Total: 5 VecDense allocations per forward pass**

**New API (Output Parameters):**
- Serial composition allocates 1 intermediate VecDense for chaining
- All other operations reuse the destination buffer
- **Total: 1 VecDense allocation per forward pass**
- **Savings: ~80% reduction**

### Backward Pass (D method calls)

**Old API (Return Values):**
Each layer.D() call returns (dYdX, dYdH):
- Perceptron(32->16):
  - dYdX: Dense(16x32) = 512 float64s
  - dYdH: Dense(16x512) = 8,192 float64s
- Bias(16):
  - dYdX: Dense(16x16) = 256 float64s
  - dYdH: Dense(16x16) = 256 float64s
- Sigmoid(16):
  - dYdX: Dense(16x16) = 256 float64s
  - dYdH: nil
- Perceptron(16->32):
  - dYdX: Dense(32x16) = 512 float64s
  - dYdH: Dense(32x512) = 16,384 float64s
- Bias(32):
  - dYdX: Dense(32x32) = 1,024 float64s
  - dYdH: Dense(32x32) = 1,024 float64s
- Serial composition: Multiple intermediate Dense matrices for chain rule
- **Total: ~10+ Dense matrix allocations, ~28,000 float64s per backward pass**

**New API (Output Parameters):**
- Pre-allocated dYdX and dYdW Dense matrices are reused
- Intermediate chain rule computations still allocate (room for future optimization)
- **Significantly reduced, especially in repeated training iterations**

## Training Loop Impact

For a typical training scenario:
- Batch size: 32
- Iterations: 10,000
- Training duration: 3 minutes

**Old API:**
- Forward allocations: 5 vectors × 32 examples × 10,000 iterations = 1,600,000 allocations
- Backward allocations: ~10 matrices × 32 examples × 10,000 iterations = 3,200,000 allocations
- **Total: ~4.8 million allocations**

**New API (with future optimization):**
- Forward allocations: 1 vector × 32 examples × 10,000 iterations = 320,000 allocations
- Backward allocations: Can be reduced to reusing 2-3 pre-allocated buffers per goroutine
- **Potential total: ~320,000 - 640,000 allocations**
- **Potential savings: 85-93% reduction**

## Current Implementation Status

The refactoring is complete for the API layer. The actual allocation savings in the current implementation are:

1. ✅ **Forward pass**: ~80% reduction achieved
2. ⚠️ **Backward pass**: Partial reduction achieved
   - Serial/Parallel composition still create temporary matrices for chain rule
   - Future optimization: could use a matrix pool or pre-allocated workspace

## Measuring Actual Savings

To measure the actual improvement, you can:

```bash
# Create a benchmark comparing iterations
go test -bench=BenchmarkTrain -benchmem -benchtime=10000x

# Use pprof to analyze allocations
go test -bench=BenchmarkTrain -memprofile=mem.prof
go tool pprof -alloc_objects mem.prof
```

## Additional Optimization Opportunities

1. **Matrix pools**: Use sync.Pool for frequently allocated intermediates
   ```go
   var densePool = sync.Pool{
       New: func() interface{} { return &mat.Dense{} },
   }
   ```

2. **Workspace pattern**: Pass a reusable workspace to D() method
   ```go
   type Workspace struct {
       temp1, temp2 *mat.Dense
   }
   D(dYdX, dYdH *mat.Dense, x mat.Vector, h []float64, ws *Workspace)
   ```

3. **Batch-aware API**: Process entire batch at once with pre-allocated buffers
   ```go
   FBatch(dst []mat.VecDense, xs []mat.Vector, h []float64)
   ```

## Conclusion

The API refactoring enables significant memory allocation reductions:
- ✅ Core infrastructure is in place
- ✅ Tests pass, examples work
- 🎯 Next step: Add benchmarks to quantify actual savings
- 🎯 Future: Optimize serial/parallel chain rule computations
