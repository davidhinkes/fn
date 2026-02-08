package layers

import (
	"math"

	"fn"
	"fn/matrixpool"

	"gonum.org/v1/gonum/mat"
)

type softmax struct {
	n int
}

func Softmax() fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return softmax{n: inputs}
	}
}

func (s softmax) Shape() (inputs, outputs, weights int) {
	return s.n, s.n, 0
}

func (s softmax) F(dst *mat.Dense, X mat.Matrix, _ []float64) {
	rows, _ := X.Dims()
	for row := 0; row < rows; row++ {
		// Find max for numerical stability.
		m := X.At(row, 0)
		for i := 1; i < s.n; i++ {
			if v := X.At(row, i); v > m {
				m = v
			}
		}
		var sum float64
		for i := 0; i < s.n; i++ {
			e := math.Exp(X.At(row, i) - m)
			dst.Set(row, i, e)
			sum += e
		}
		for i := 0; i < s.n; i++ {
			dst.Set(row, i, dst.At(row, i)/sum)
		}
	}
}

func (s softmax) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	// Recompute forward output for the VJP.
	Y := matrixpool.GetDense(rows, s.n)
	defer matrixpool.PutDense(Y)
	s.F(Y, X, h)

	for row := 0; row < rows; row++ {
		// Jacobian: dY[i]/dX[j] = y[i] * (delta_{ij} - y[j])
		// VJP: dLdX[j] = y[j] * (dLdY[j] - dot(dLdY, y))
		var dot float64
		for i := 0; i < s.n; i++ {
			dot += dLdY.At(row, i) * Y.At(row, i)
		}
		for j := 0; j < s.n; j++ {
			dLdX.Set(row, j, Y.At(row, j)*(dLdY.At(row, j)-dot))
		}
	}
	// dLdH is nil - softmax has no weights
}
