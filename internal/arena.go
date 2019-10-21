package arena

import (
	"github.com/davidhinkes/slicearena"
	"gonum.org/v1/gonum/mat"
)

type T struct {
	arena *slicearena.T
}

func Make() T {
	return T{
		arena: slicearena.New(float64(0)),
	}
}

func (t T) Reset() {
	t.arena.Reset()
}

func (t T) NewDense(r, c int) (*mat.Dense, []float64) {
	s := t.arena.MakeSlice(r*c).([]float64)
	return mat.NewDense(r,c, s),s
}

func (t T) NewVecDense(r int) (*mat.VecDense, []float64) {
	s := t.arena.MakeSlice(r).([]float64)
	return mat.NewVecDense(r, s),s
}