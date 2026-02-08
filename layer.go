package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	// F is the layer's forward function. X is (B×n) where B is batch size
	// and n is the input dimension. dst must be pre-sized to (B×m) where m
	// is the output dimension from Shape().
	F(dst *mat.Dense, X mat.Matrix, h []float64)

	// D computes gradients via backpropagation (Vector-Jacobian Product).
	// dLdY is (B×m), the upstream gradient of loss w.r.t. this layer's output.
	// Computes dLdX (B×n), the gradient w.r.t. input, and dLdH, the gradient
	// w.r.t. weights summed over the batch. dLdX must be pre-sized to (B×n),
	// dLdH to numWeights (from Shape()). dLdH may be nil if the layer has no weights.
	D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64)

	// Shape returns the per-example dimensions of this layer.
	// inputs (n) is the column count of X in F and D.
	// outputs (m) is the column count of dst in F and dLdY in D.
	// weights is the length of the h slice.
	Shape() (inputs, outputs, weights int)
}

type LayerBuilder func(inputs int) Layer
