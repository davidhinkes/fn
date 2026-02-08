package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

type scalar struct {
	n int
}

func Scalar() fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return scalar{
			n: inputs,
		}
	}
}

func (s scalar) Shape() (inputs, outputs, weights int) {
	return s.n, s.n, s.n
}

func (s scalar) F(dst *mat.Dense, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	// Y_bi = X_bi * h_i (broadcast element-wise multiply)
	for row := 0; row < rows; row++ {
		for i := 0; i < s.n; i++ {
			dst.Set(row, i, X.At(row, i)*h[i])
		}
	}
}

func (s scalar) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	// dLdX_bi = dLdY_bi * h_i
	// dLdH_i = Σ_b (dLdY_bi * X_bi)
	for i := 0; i < s.n; i++ {
		var sum float64
		for row := 0; row < rows; row++ {
			dLdX.Set(row, i, dLdY.At(row, i)*h[i])
			sum += dLdY.At(row, i) * X.At(row, i)
		}
		dLdH.SetVec(i, sum)
	}
}
