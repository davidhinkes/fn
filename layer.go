package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	// F is the layer's forward function. Given vector x as an input, fills dst with the output.
	// dst will be resized if needed. |dst| = |y|
	F(dst *mat.VecDense, x mat.Vector, h []float64)

	// D computes the partial derivatives of the layer.
	// dYdX and dYdH are filled with the derivatives. They will be resized if needed.
	// If dYdH is passed as nil, derivatives with respect to weights are not computed.
	// For layers with no weights, dYdH will be left unmodified (caller can pass nil).
	D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64)

	NumWeights() int
}
