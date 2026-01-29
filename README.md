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

### Key Learnings

1. **Performance intuition can be wrong**: DiagDense *should* be faster but isn't - measure everything

2. **API stability is hard**: Multiple major API shakeups before finding the right abstractions

3. **Simplicity wins**: Removed arena optimization, simplified training loop, removed dead code

4. **Numerical stability matters**: Multiple commits fixing overflow, underflow, and gradient issues

5. **Observability is crucial**: Gradient norm tracking revealed training issues

6. **Testing prevents regression**: Extensive layer tests catch numerical problems early

7. **Memory management trade-offs**: Started with custom arena, removed it - simpler is often better

8. **Gradient descent is subtle**: Multiple iterations on step size, normalization, and clipping

