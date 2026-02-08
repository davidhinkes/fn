package layers

import (
	"math"

	"fn"
	"fn/matrixpool"

	"gonum.org/v1/gonum/floats"
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
		dstRow := dst.RawRowView(row)
		// Find max for numerical stability.
		m := X.At(row, 0)
		for i := 1; i < s.n; i++ {
			if v := X.At(row, i); v > m {
				m = v
			}
		}
		var sum float64
		for i := range dstRow {
			e := math.Exp(X.At(row, i) - m)
			dstRow[i] = e
			sum += e
		}
		floats.Scale(1/sum, dstRow)
	}
}

func (s softmax) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	// Recompute forward output for the VJP.
	Y := matrixpool.GetDense(rows, s.n)
	defer matrixpool.PutDense(Y)
	s.F(Y, X, h)

	for row := 0; row < rows; row++ {
		// VJP: dLdX[j] = y[j] * (dLdY[j] - dot(dLdY, y))
		yRow := Y.RawRowView(row)
		dLdXRow := dLdX.RawRowView(row)
		// dLdY is mat.Matrix — read into dLdXRow as scratch, then compute in-place.
		for i := range dLdXRow {
			dLdXRow[i] = dLdY.At(row, i)
		}
		dot := floats.Dot(dLdXRow, yRow)
		for j := range dLdXRow {
			dLdXRow[j] = yRow[j] * (dLdXRow[j] - dot)
		}
	}
	// dLdH is nil - softmax has no weights
}
