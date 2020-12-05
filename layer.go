package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	// F is the layer's forward function. Given vector x as an input, returns an output vector.
	// |return| = |y|
	F(x mat.Vector) mat.Vector

	// D returns the partial derivitives of the layer.
	D(x mat.Vector) (dYdX mat.Matrix, dYdH mat.Matrix)

	// Hyperparameters returns the underlying hyperparemeters, which
	// will be modified.
	Hyperparameters() *mat.VecDense
}
