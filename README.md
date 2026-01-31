# FN

FN is a functional network, a generalization of neural networks implemented in Go.

## Features

- **Layer composition**: Serial and parallel layer composition for building complex network architectures
- **Multiple layer types**: Perceptron, ReLU, sigmoid, bias, scalar, radial basis functions, and static function layers
- **Training**: Multi-threaded batch training with gradient descent
- **Persistence**: YAML marshaling for model serialization
- **Cloud integration**: Google Cloud Storage support for model storage

## Tips

### Building via GCP

> gcloud builds submit -t us-docker.pkg.dev/enhanced-kit-571/fn-dev/fn

## Future Improvements

### Cross-Entropy Loss for Classification

**Current state**: The codebase uses Squared Error Loss (MSE) for both regression and classification tasks. While MSE works fine for regression, it's suboptimal for classification.

**Why Cross-Entropy is better for classification**:
1. **Probabilistically motivated** - Measures KL divergence between true and predicted probability distributions
2. **Better gradient behavior** - Gradients remain strong even when predictions are wrong, leading to faster convergence
3. **Natural pairing with Softmax** - The combination softmax + cross-entropy has a clean gradient: dL/dLogits = yHat - y
4. **Avoids gradient saturation** - MSE + sigmoid can have vanishing gradients when predictions are confidently wrong

**Implementation approach**:
1. Add new loss function in `lossfunctions/cross_entropy.go`:
   - Implement `F(y, yHat)` returning: `-sum(y * log(yHat))` and gradient `-(y / yHat)`
2. Add Softmax activation layer in `layers/softmax.go`:
   - Forward: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`
   - Backward: Jacobian matrix for chain rule
3. OR: Combine Softmax + Cross-Entropy into single `SoftmaxCrossEntropy` loss for numerical stability and cleaner gradients
4. Update `cmd/binary_integer_example/main.go` to use new loss function

**Trade-offs**: Adds complexity to the loss function interface, but significantly improves classification training performance.

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

