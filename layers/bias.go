package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

type bias struct {
	n        int
	identity mat.Matrix
}

func MakeBiasLayer(n int) fn.Layer {
	ident := make([]float64, n)
	for i := range ident {
		ident[i] = 1.
	}
	b := bias{
		n:        n,
		identity: diagFromSlice(ident),
	}
	return b
}

func (b bias) NumWeights() int {
	return b.n
}

func (b bias) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	w := mat.NewVecDense(b.n, h)
	dst.AddVec(x, w)
}

func (b bias) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, _ []float64) {
	dYdX.CloneFrom(b.identity)
	if dYdH != nil {
		dYdH.CloneFrom(b.identity)
	}
}
