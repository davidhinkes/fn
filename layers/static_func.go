package layers

import (
	"gonum.org/v1/gonum/mat"
)

// staticFunc is a fn.Layer that is based on a static function, which the
// caller can inject. This is intended to be a utility type used to more
// easily create simple layers w/o boilerplate.
type staticFunc struct {
	f func(x float64) float64
	d func(x float64) float64
}

func (s staticFunc) F(x mat.Vector, _ []float64) mat.Vector {
	y := make([]float64, x.Len())
	for i, _ := range y {
		y[i] = s.f(x.AtVec(i))
	}
	return mat.NewVecDense(len(y), y)
}

func (s staticFunc) D(x mat.Vector, _ []float64) (mat.Matrix, mat.Matrix) {
	v := make([]float64, x.Len())
	for i := range v {
		v[i] = s.d(x.AtVec(i))
	}
	return diagFromSlice(v), nil
}

func (_ staticFunc) NumWeights() int {
	return 0
}
