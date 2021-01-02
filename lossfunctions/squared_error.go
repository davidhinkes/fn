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

func (s squaredError) F(y mat.Vector, yHat mat.Vector) (float64, mat.Vector) {
	e := mat.NewVecDense(y.Len(), nil)
	e.SubVec(y, yHat)
	d := mat.NewVecDense(y.Len(), nil)
	d.ScaleVec(2.0, e)
	return mat.Dot(e, e), d
}
