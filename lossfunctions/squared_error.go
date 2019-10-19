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
	e := &mat.VecDense{}
	e.SubVec(y,yHat)
	e.MulElemVec(e,e)
	return mat.Sum(e), s.d(y, yHat)
}

func (s squaredError) d(y mat.Vector, yHat mat.Vector) mat.Vector {
	e := &mat.VecDense{}
	e.SubVec(y, yHat)
	e.ScaleVec(2.0, e)
	return e
}