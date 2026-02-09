package layers

import (
	"gonum.org/v1/gonum/mat"
)

// staticFunc is a VectorLayer based on a static element-wise function.
type staticFunc struct {
	f func(x float64) float64
	d func(x float64) float64
	n int
}

func (s staticFunc) F(dst *mat.VecDense, x mat.Vector, _ []float64) {
	for i := 0; i < s.n; i++ {
		dst.SetVec(i, s.f(x.AtVec(i)))
	}
}

func (s staticFunc) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, _ []float64) {
	for i := 0; i < s.n; i++ {
		dLdX.SetVec(i, dLdY.AtVec(i)*s.d(x.AtVec(i)))
	}
}

func (s staticFunc) Shape() (inputs, outputs, weights int) {
	return s.n, s.n, 0
}
