package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

func Scalar() fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return wrapVectorLayer(scalar{
			n: inputs,
		})
	}
}

type scalar struct {
	n int
}

// We expect scalar to adhere to the VectorLayer
var _ VectorLayer = scalar{}

func (s scalar) Shape() (inputs, outputs, weights int) {
	return s.n, s.n, s.n
}

func (s scalar) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	for i := 0; i < s.n; i++ {
		dst.SetVec(i, x.AtVec(i)*h[i])
	}
}

func (s scalar) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64) {
	for i := 0; i < s.n; i++ {
		dLdX.SetVec(i, dLdY.AtVec(i)*h[i])
		dLdH.SetVec(i, dLdY.AtVec(i)*x.AtVec(i))
	}
}
