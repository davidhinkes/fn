package layers

import (
	"fn"
	"fn/matrixpool"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type bias struct {
	n int
}

func Bias() fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return bias{n: inputs}
	}
}

func (b bias) Shape() (inputs, outputs, weights int) {
	return b.n, b.n, b.n
}

func (b bias) F(dst *mat.Dense, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	dst.Copy(X)
	for row := 0; row < rows; row++ {
		floats.Add(dst.RawRowView(row), h)
	}
}

func (b bias) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, _ []float64) {
	rows, _ := dLdY.Dims()
	dLdX.Copy(dLdY)
	// dLdH = column sums of dLdY = dLdYᵀ · 1
	ones := matrixpool.GetVec(rows)
	defer matrixpool.PutVec(ones)
	for i := range ones.RawVector().Data {
		ones.RawVector().Data[i] = 1
	}
	dLdH.MulVec(dLdY.T(), ones)
}
