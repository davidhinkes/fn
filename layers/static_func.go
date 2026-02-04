package layers

import (
	"log"

	"gonum.org/v1/gonum/mat"
)

// staticFunc is a fn.Layer that is based on a static function, which the
// caller can inject. This is intended to be a utility type used to more
// easily create simple layers w/o boilerplate.
type staticFunc struct {
	f func(x float64) float64
	d func(x float64) float64
	n int
}

func (s staticFunc) F(dst *mat.VecDense, x mat.Vector, _ []float64) {
	if x.Len() != s.n {
		log.Fatalf("staticFunc expecting input of size %v, got input %v", x.Len(), s.n)
	}
	for i := 0; i < x.Len(); i++ {
		dst.SetVec(i, s.f(x.AtVec(i)))
	}
}

func (s staticFunc) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, _ []float64) {
	// Element-wise: dLdX[i] = dLdY[i] * d(x[i])
	// This is O(n) instead of O(n²) matrix multiplication
	for i := range x.Len() {
		dLdX.SetVec(i, dLdY.AtVec(i)*s.d(x.AtVec(i)))
	}
	// dLdH is nil - static functions have no weights
}

func (s staticFunc) Shape() (inputs, outputs, weights int) {
	return s.n, s.n, 0
}
