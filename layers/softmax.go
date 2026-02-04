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

func (s softmax) F(dst *mat.VecDense, x mat.Vector, _ []float64) {
	n := x.Len()
	// Find max for numerical stability.
	m := x.AtVec(0)
	for i := 1; i < n; i++ {
		v := x.AtVec(i)
		if v < m {
			continue
		}
		m = v
	}
	var sum float64
	for i := 0; i < n; i++ {
		e := math.Exp(x.AtVec(i) - m)
		dst.SetVec(i, e)
		sum += e
	}
	dst.ScaleVec(1.0/sum, dst)
}

func (s softmax) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64) {
	n := x.Len()
	y := matrixpool.GetVec(n)
	defer matrixpool.PutVec(y)
	s.F(y, x, h)
	// Jacobian: dY[i]/dX[j] = y[i] * (delta_{ij} - y[j])
	// VJP: dLdX[j] = sum_i dLdY[i] * dY[i]/dX[j]
	//             = y[j] * (dLdY[j] - dot(dLdY, y))
	dot := mat.Dot(dLdY, y)
	for j := 0; j < n; j++ {
		dLdX.SetVec(j, y.AtVec(j)*(dLdY.AtVec(j)-dot))
	}
	// dLdH is nil - softmax has no weights
}
