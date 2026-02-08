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
	n int
}

func (s staticFunc) F(dst *mat.Dense, X mat.Matrix, _ []float64) {
	rows, _ := X.Dims()
	for row := 0; row < rows; row++ {
		for i := 0; i < s.n; i++ {
			dst.Set(row, i, s.f(X.At(row, i)))
		}
	}
}

func (s staticFunc) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, _ []float64) {
	rows, _ := X.Dims()
	// Element-wise: dLdX[b,i] = dLdY[b,i] * d(X[b,i])
	for row := 0; row < rows; row++ {
		for i := 0; i < s.n; i++ {
			dLdX.Set(row, i, dLdY.At(row, i)*s.d(X.At(row, i)))
		}
	}
	// dLdH is nil - static functions have no weights
}

func (s staticFunc) Shape() (inputs, outputs, weights int) {
	return s.n, s.n, 0
}
