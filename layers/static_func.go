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

func (s staticFunc) F(dst *mat.VecDense, x mat.Vector, _ []float64) {
	dst.ReuseAsVec(x.Len())
	for i := 0; i < x.Len(); i++ {
		dst.SetVec(i, s.f(x.AtVec(i)))
	}
}

func (s staticFunc) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, _ []float64) {
	v := make([]float64, x.Len())
	for i := range v {
		v[i] = s.d(x.AtVec(i))
	}
	diag := diagFromSlice(v)
	dYdX.CloneFrom(diag)
	// dYdH is not modified - static functions have no weights
}

func (_ staticFunc) NumWeights() int {
	return 0
}
