package fn

import (
	"gonum.org/v1/gonum/mat"
)

type LossFunction interface {
	// F computes the loss and writes the partial derivative with
	// respect to y into dst.
	F(dst *mat.VecDense, y mat.Vector, yHat mat.Vector) float64
}
