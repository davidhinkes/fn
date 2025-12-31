package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	// F is the layer's forward function. Given vector x as an input, returns an output vector.
	// |return| = |y|
	F(x mat.Vector, h []float64) mat.Vector

	// D returns the partial derivatives of the layer.
	D(x mat.Vector, h []float64) (dYdX mat.Matrix, dYdH mat.Matrix)

	NumWeights() int
}
