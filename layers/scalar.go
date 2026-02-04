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

func (s scalar) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	dst.MulElemVec(x, mat.NewVecDense(s.n, h))
}

func (s scalar) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64) {
	// y[i] = x[i] * h[i]
	// dLdX[i] = dLdY[i] * h[i]
	// dLdH[i] = dLdY[i] * x[i]
	dLdX.MulElemVec(dLdY, mat.NewVecDense(len(h), h))
	dLdH.MulElemVec(dLdY, x)
}
