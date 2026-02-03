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

- **`Layer`** — forward pass `F(dst, x, h)`, derivatives `D(dYdX, dYdH, x, h)`, and `Shape() (inputs, outputs, weights)`. All methods use output parameters (`dst`, `dYdX`, `dYdH`) that callers must pre-allocate to the correct dimensions from `Shape()`. The weight vector `h` is a flat `[]float64` slice owned by the `Model`; layers interpret their slice segment.
- **`LayerBuilder`** — `func(inputs int) Layer`. Enables automatic dimension threading in serial composition—each builder receives the output dimension of the preceding layer.
- **`LossFunction`** — `F(dst, y, yHat) float64` computes loss and writes gradient into `dst`.
- **`Model`** — holds a single composed `Layer` and a flat weight vector. `MakeModel(inputs, builders...)` constructs via `SerialBuilder`. Training is multi-threaded: one goroutine per example in the batch, gradients accumulated via channel.

### Composition

- **`Serial(layers...)`** — chains layers via recursive `serialNode` binary tree. Weights are partitioned by slicing `h[:leftWeights]` / `h[leftWeights:]`. The `D` method implements the chain rule with matrix multiplication and horizontal augmentation of weight gradients.
- **`Parallel(layers...)`** — runs layers independently on the same input, concatenates outputs. Derivatives are block-diagonal for dYdX and block-structured for dYdH.

### Layer Implementations (package `layers`)

Layers with weights: `Perceptron(outputs)`, `Bias()`, `Scalar()`, `Radial()`. Weightless layers: `Sigmoid()`, `ReLU()` (both via `staticFunc`). All return `LayerBuilder`.

### Memory Management (package `matrixpool`)

Dimension-keyed `sync.Pool` for `*mat.Dense` and `*mat.VecDense`. Used throughout `serial.go` and `train.go` to avoid allocations in hot paths. Pooled matrices are **not zeroed** on `Get`—callers must fully overwrite or explicitly `Zero()`.

### Testing (package `test`)

`Truth` interface defines a ground-truth function with `Dims()`, `F(dst, x)`, and `Rand(dst)`. `MakeExamples` generates training data. Tests in `layers/` use `testLayer()` which trains to convergence and fails if error stays above threshold.

## Key Design Decisions

- **Output parameters everywhere**: `F` and `D` write into caller-provided buffers rather than returning new allocations. This enables pooling.
- **Flat weight vector**: All weights for a composed model live in a single `[]float64`. Layers receive slices. This allows the model to update weights directly via `AddScaledVec`.
- **DiagDense avoidance**: gonum's `DiagDense` causes 450x slowdown in `Dense.Mul()`. The codebase uses regular `Dense` matrices for diagonal data (see `layers/diag.go`).
