# FN

FN is a functional network, a generalization of neural networks implemented in Go.

## Features

- **Batch-oriented Layer interface**: Forward and backward passes operate on full batch matrices, enabling level-3 BLAS (GEMM) for layers like Perceptron
- **VectorLayer abstraction**: Simpler per-vector interface for element-wise and pointwise layers, automatically adapted to the batch interface via `vectorLayerConverter`
- **Layer composition**: Serial composition for chaining layers sequentially
- **Multiple layer types**: Perceptron, ReLU, sigmoid, softmax, bias, scalar, radial basis functions, and static function layers
- **VJP-based backpropagation**: Backward pass uses Vector-Jacobian Products instead of materializing full Jacobian matrices — O(n) for element-wise layers instead of O(n^2)
- **Training**: Sequential batch training with gradient descent
- **Loss functions**: Squared error and cross-entropy
- **Persistence**: YAML marshaling for model serialization
- **Cloud integration**: Google Cloud Storage support for model storage

## Tips

### Building via GCP

> gcloud builds submit -t us-docker.pkg.dev/enhanced-kit-571/fn-dev/fn

## Future Improvements

### Fused Softmax + Cross-Entropy

Softmax and cross-entropy are both implemented as separate components (`layers/softmax.go`, `lossfunctions/cross_entropy.go`). A fused `SoftmaxCrossEntropy` loss could provide better numerical stability and the clean gradient `dL/dLogits = yHat - y`, avoiding the per-element division in the separate backward passes.

## Project History & Key Learnings

### Development Phases

#### Phase 1: Foundation (Oct 2019 - Early 2020)
- Initial implementation with basic components: perceptron layer, squared error loss, training algorithm
- Early memory arena optimization experiments
- Foundation of loss function API and training mechanics
- **Bug fix**: Squared error partial derivative correction - an important early numerical bug

#### Phase 2: API Evolution & Parallelization (Mid 2020 - 2021)
- Multiple major API refactorings as the architecture matured
- **Critical architectural decision**: Removed arena-based memory management - the optimization wasn't worth the complexity
- **Key API change**: Layers no longer retain partial derivatives, enabling cleaner parallelization
- Implemented multi-threaded training
- **Terminology shift**: "hyperparameters" → "weights" for more accurate naming
- Simplified model by removing `Trainer` struct

#### Phase 3: Composition & Serialization (2021)
- **Major addition**: Serial function composition for chaining layers together
- Added parallel layer support for concurrent execution
- Implemented YAML marshaling for model persistence
- Google Cloud Storage integration for model storage
- Kubernetes deployment configuration

#### Phase 4: Layer Expansion (2021-2022)
- Added multiple layer types: ReLU, bias, scalar, sigmoid, radial basis functions
- Created `static_func` layer for simple function-based layers
- Developed comprehensive test infrastructure
- Created example programs: binary_integer_example and regression_example

#### Phase 5: Serial Algorithm Refinement (Jan 2026)
- Experimented with recursive serial composition
- Removed dead serialv1 implementation after finding better approach
- Used Unicode symbols (ℵ and ℶ - Aleph and Beth) for mathematical notation

#### Phase 6: Performance Crisis & Resolution (Jan 2026)
**The DiagDense Performance Disaster:**
- Discovered **450x+ performance degradation** from using `DiagDense` matrices
- **Root cause**: While `DiagDense` seems like the right choice for diagonal matrices, matrix multiplication with it is catastrophically slow
- **Solution**: Use regular `Dense` matrices for diagonal operations despite seeming wasteful
- Added extensive documentation explaining this non-intuitive trade-off

**Training Algorithm Improvements:**
- Revamped gradient descent step size calculation
- Fixed numerical stability by summing losses before normalization
- Removed unnecessary gradient normalization that was hindering learning
- Added gradient clipping (clip value = 5.0) to prevent exploding gradients

**Numerical Stability:**
- **Sigmoid overflow/underflow fix**: Branching on sign of input to keep exp() operations safe for |x| > 100
- Improved test coverage to catch numerical instabilities

#### Phase 7: Observability (Jan 2026)
- Added gradient norm tracking to monitor training health
- Improved logging and status callbacks
- Added animated spinner display to reduce screen clutter

#### Phase 8: Allocation Optimization & API Redesign (Jan 2026)
**The Output Parameter Revolution:**
- **Major API change**: Transitioned from allocation-on-return to output parameters
  - `F(dst *mat.VecDense, x mat.Vector, h []float64)` instead of `F(x, h) mat.Vector`
  - `D(dYdX, dYdH *mat.Dense, x, h)` instead of `D(x, h) (mat.Matrix, mat.Matrix)`
- **Shape() API**: Replaced `NumWeights()` with `Shape() (inputs, outputs, weights int)`
  - Eliminated runtime dimension queries
  - Enabled proper pre-allocation of derivative matrices

**Matrix Pooling with sync.Pool:**
- Implemented `matrixpool` package for temporary matrix reuse
- Pooled allocations in hot paths: `serial.go`, `train.go`, layer implementations
- **Critical insight**: Most temporary matrices in backpropagation can be pooled

**diag Function Optimization:**
- Changed diagonal matrix functions to use output parameters
- `diag(dst *mat.Dense, x)` instead of `diag(x) mat.Matrix`
- Eliminated allocations in activation function derivatives

**Performance Results (destapi branch vs main):**
- **Speed**: 15-38% faster across all operations
  - Serial backward: 17% faster
  - Parallel backward: 37.6% faster (best improvement)
  - Training step: 14.5% faster
- **Memory**: 41-95% reduction in allocations
  - Serial forward: 94.6% less (2,368 → 128 bytes, 80% fewer allocs)
  - Serial backward: 63.7% less (12.8 MB → 4.6 MB)
  - Training step: 83.8% less (26 MB → 4.2 MB, 16.9% fewer allocs)
- **GC pressure**: Dramatically reduced due to matrix pooling

**Key Implementation Details:**
- Used `defer` pattern for pool returns to ensure cleanup
- Zero'd pooled matrices before reuse to avoid stale data
- Maintained thread-safety with mutex-protected pool maps keyed by matrix dimensions

#### Phase 9: Batch Layer Interface & VJP (Jan-Feb 2026)
**The Batch-First Rewrite:**
- **Major API change**: Layer `F` and `D` now operate on batch matrices (`mat.Matrix` inputs, `*mat.Dense` outputs) instead of single vectors
- Perceptron forward `Y = X·Wᵀ` and backward `dLdW = dLdYᵀ·X` are single GEMM calls — level-3 BLAS replaces B separate matrix-vector multiplies
- Inputs use `mat.Matrix` interface (read-only); outputs use `*mat.Dense` (caller pre-allocates)
- `dLdH` stays `*mat.VecDense` since the model owns one flat weight vector regardless of batch size

**VJP over Jacobians:**
- Replaced full Jacobian materialization with Vector-Jacobian Products in the backward pass
- Element-wise layers (sigmoid, ReLU, softmax) went from O(n^2) diagonal Jacobian multiply to O(n) pointwise
- Eliminated gonum's slow `DiagDense` multiplication entirely

**Softmax & Cross-Entropy:**
- Implemented softmax with numerically stable max-subtraction trick
- Added cross-entropy loss function
- Softmax VJP: `dLdX[j] = y[j] * (dLdY[j] - dot(dLdY, y))` — no Jacobian needed

**Sequential Training:**
- Removed goroutines from training loop — batch GEMM already parallelizes via BLAS threads
- Simplified `Model.Train` to a single forward/backward call on the full batch matrix

#### Phase 10: VectorLayer Abstraction (Feb 2026)
**Two-tier Layer design:**
- Added `VectorLayer` interface — per-vector `F`/`D` signatures using `mat.Vector` instead of `mat.Matrix`
- `vectorLayerConverter` adapter wraps any `VectorLayer` to satisfy the batch `fn.Layer` interface
- Layers that don't benefit from batch GEMM (sigmoid, ReLU, softmax, scalar, radial, bias) implement the simpler `VectorLayer`; the adapter handles batch iteration
- Perceptron stays as a direct `fn.Layer` since it genuinely benefits from GEMM

**Branchless row extraction:**
- Used `mat.Row(buf, i, M)` to extract rows from any `mat.Matrix` into pooled buffers — works uniformly for `*mat.Dense` and `mat.Transpose` (from single-vector `Eval`) without type assertions
- Scratch vectors pooled once before the loop and reused across iterations

### Key Learnings

1. **Performance intuition can be wrong**: DiagDense *should* be faster but isn't - measure everything

2. **API stability is hard**: Multiple major API shakeups before finding the right abstractions

3. **Simplicity wins**: Removed arena optimization, simplified training loop, removed dead code

4. **Numerical stability matters**: Multiple commits fixing overflow, underflow, and gradient issues

5. **Observability is crucial**: Gradient norm tracking revealed training issues

6. **Testing prevents regression**: Extensive layer tests catch numerical problems early

7. **Memory management trade-offs**: Started with custom arena, removed it - now using sync.Pool for targeted pooling

8. **Gradient descent is subtle**: Multiple iterations on step size, normalization, and clipping

9. **Output parameters beat return allocations**: For hot paths, pre-allocating and reusing memory dramatically reduces GC pressure

10. **Allocation profiling is essential**: Benchmarking revealed that 83-95% of training memory was temporary allocations

11. **The right abstraction enables optimization**: Shape() API made pre-allocation possible; output parameters enabled pooling

12. **sync.Pool is powerful when used correctly**: Dimension-keyed pools with Zero() before reuse provides safe, fast matrix reuse

13. **VJP beats full Jacobians**: For element-wise layers, computing the Vector-Jacobian Product directly is O(n) vs O(n^2) for materializing the diagonal Jacobian and multiplying — and avoids gonum's slow `DiagDense`

14. **Batch GEMM subsumes thread-per-example parallelism**: Once the Layer interface operates on batch matrices, BLAS threads parallelize the GEMM internally — explicit goroutines in the training loop become redundant overhead

15. **Two-tier interfaces reduce boilerplate**: Most layers are naturally per-vector; forcing them into a batch interface adds repetitive row-iteration code. A simple adapter (`VectorLayer` → `fn.Layer`) lets each layer be written at its natural level of abstraction while the batch plumbing is written once

16. **`mat.Row` unifies row extraction**: Rather than type-asserting `mat.Matrix` to `*mat.Dense` for `RowViewOf` with a fallback for non-Dense, `mat.Row(buf, i, M)` works uniformly on any `mat.Matrix` — one code path, no branching

