package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	// F is the layer's forward function. Given vector x as an input, fills dst with the output.
	// dst must be pre-sized to the output dimension from Shape().
	F(dst *mat.VecDense, x mat.Vector, h []float64)

	// D computes gradients via backpropagation (Vector-Jacobian Product).
	// Given the upstream gradient dLdY (gradient of loss w.r.t. this layer's output),
	// computes dLdX (gradient w.r.t. input) and dLdH (gradient w.r.t. weights).
	// dLdX must be pre-sized to inputs, dLdH to weights (from Shape()).
	// dLdH may be nil if the layer has no weights.
	D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64)

	// Shape returns the layer dimensions: (inputs, outputs, numWeights).
	Shape() (inputs, outputs, weights int)
}

type LayerBuilder func(inputs int) Layer
