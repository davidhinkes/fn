package lossfunctions

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

func NewSquaredError() fn.LossFunction {
	return squaredError{}
}

type squaredError struct {
}

func (s squaredError) F(dst *mat.VecDense, y mat.Vector, yHat mat.Vector) float64 {
	// We're being space efficient by re-using dst as a temp vector.
	// The order of these commands is very important.
	dst.SubVec(y, yHat)
	loss := mat.Dot(dst, dst)
	dst.ScaleVec(2.0, dst)
	return loss
}
