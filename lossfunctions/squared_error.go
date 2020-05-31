package lossfunctions

import (
	"fn"
	arena "fn/internal"

	"gonum.org/v1/gonum/mat"
)

func NewSquaredError() fn.LossFunction {
	return squaredError{
		arena: arena.Make(),
	}
}

type squaredError struct {
	arena arena.T
}

func (s squaredError) F(y mat.Vector, yHat mat.Vector) (float64, mat.Vector) {
	s.arena.Reset()
	e, _ := s.arena.NewVecDense(y.Len())
	e.SubVec(y, yHat)
	loss, _ := s.arena.NewVecDense(y.Len())
	loss.MulElemVec(e, e)
	d, _ := s.arena.NewVecDense(y.Len())
	d.ScaleVec(2.0, e)
	return mat.Sum(loss), d
}
