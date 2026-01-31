package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	// F is the layer's forward function. Given vector x as an input, fills dst with the output.
	// dst must be pre-sized to the output dimension from Shape().
	F(dst *mat.VecDense, x mat.Vector, h []float64)

	// D computes the partial derivatives of the layer.
	// dYdX must be pre-sized to (outputs, inputs) and dYdH to (outputs, weights) from Shape().
	D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64)

	// Shape returns the layer dimensions: (inputs, outputs, numWeights).
	Shape() (inputs, outputs, weights int)
}

type LayerBuilder func(inputs int) Layer
