package layers

import (
	"fn"

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

func (b bias) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	w := mat.NewVecDense(b.n, h)
	dst.AddVec(x, w)
}

func (b bias) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, _ []float64) {
	// y = x + b, so dLdX & dLdH are both just equal to dLdY
	// dLdY is known
	// dLdX = dLdY * dYdX; dLdY is I
	// dLdH = dLdY * dYdH; dYdH is I
	dLdX.CopyVec(dLdY)
	dLdH.CopyVec(dLdY)
}
