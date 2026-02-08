# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
go build ./...                          # build all packages
go test ./...                           # run all tests (layer tests train for 15s each)
go test ./layers -run TestBiasLayer     # run a single test
go test -bench=. ./...                  # run benchmarks
go test -bench=BenchmarkTrain -benchmem # benchmark with allocation reporting
```

Tests verify convergence by training models against known truth functions for up to 15 seconds with an allowable error threshold of 1e-5. They are slow by design.

## Architecture

This is a neural network framework in pure Go built on gonum for linear algebra. Models are composed from layers using a functional, composable API.

### Core Interfaces (package `fn`)

- **`Layer`** — batch-oriented forward/backward interface. `F(dst *mat.Dense, X mat.Matrix, h []float64)` computes the forward pass for a batch of inputs. `D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64)` computes gradients via Vector-Jacobian Products (VJP): given upstream gradient `dLdY`, it computes `dLdX` (gradient w.r.t. input) and `dLdH` (gradient w.r.t. weights, summed over the batch). `Shape() (inputs, outputs, weights)` returns per-example dimensions. Input parameters (`X`, `dLdY`) use the `mat.Matrix` interface (read-only); output parameters (`dst`, `dLdX`) use `*mat.Dense` (caller pre-allocates). `dLdH` stays `*mat.VecDense` since the model owns one flat weight vector regardless of batch size. The weight vector `h` is a flat `[]float64` slice owned by the `Model`; layers interpret their slice segment.
- **`LayerBuilder`** — `func(inputs int) Layer`. Enables automatic dimension threading in serial composition—each builder receives the output dimension of the preceding layer.
- **`LossFunction`** — `F(dst, y, yHat) float64` computes loss and writes gradient into `dst`. Currently per-vector; the train loop iterates over batch rows.
- **`Model`** — holds a single composed `Layer` and a flat weight vector. `MakeModel(inputs, builder)` takes a single `LayerBuilder`. `Train(X, YHat *mat.Dense, lossFunction, alpha)` takes batch inputs as Dense matrices. `Eval(dst, x)` stays single-vector for convenience (passes `x.T()` as a 1-row batch internally).

### Layer Implementations (package `layers`)

Layers with weights: `Perceptron(outputs)`, `Bias()`, `Scalar()`, `Radial()`. Weightless layers: `Sigmoid()`, `ReLU()` (both via `staticFunc`). Composition: `Serial(builders...)` chains layers sequentially. All return `LayerBuilder`.

### Memory Management (package `matrixpool`)

Dimension-keyed `sync.Pool` for `*mat.Dense` and `*mat.VecDense`. Used throughout `layers/serial.go`, `layers/perceptron.go`, and `train.go` to avoid allocations in hot paths. Pooled matrices are **not zeroed** on `Get`—callers must fully overwrite or explicitly `Zero()`.

### Testing (package `test`)

`Truth` interface defines a ground-truth function with `Dims()`, `F(dst, x)`, and `Rand(dst)`. `MakeExamples(truth, n)` returns `(*mat.Dense, *mat.Dense)` — batch matrices ready for `Train`. Tests in `layers/` use `testLayer()` which trains to convergence and fails if error stays above threshold. Gradient checking uses batch size 1 (1-row Dense) for finite differences.

## Key Design Decisions

- **Batch-first Layer interface**: `F` and `D` operate on batches of examples (`mat.Matrix` inputs, `*mat.Dense` outputs). Inputs use the `mat.Matrix` interface to clearly distinguish read-only parameters from mutable output parameters, and to accept transposed views without wrapping. Scalars use lowercase names; matrices use uppercase.
- **Output parameters everywhere**: `F` and `D` write into caller-provided buffers rather than returning new allocations. This enables pooling.
- **Flat weight vector**: All weights for a composed model live in a single `[]float64`. Layers receive slices. This allows the model to update weights directly via `AddScaledVec`.
- **VJP over Jacobians**: The backward pass uses Vector-Jacobian Products rather than materializing full Jacobian matrices. This is O(n) for element-wise layers (sigmoid, ReLU) instead of O(n²), and avoids gonum's slow `DiagDense` multiplication.
- **GEMM over GEMV**: Perceptron forward/backward uses matrix-matrix multiply (level-3 BLAS) instead of per-example matrix-vector multiply (level-2 BLAS). The weight gradient `dLdW = dLdYᵀ·X` replaces B outer products with a single GEMM.
