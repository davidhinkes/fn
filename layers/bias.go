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
	identityMatrix := mat.NewDense(n, n, nil)
	for i := range n {
		identityMatrix.Set(i, i, 1)
	}
	return bias{
		n:        n,
		identity: identityMatrix,
	}
}

func (b bias) Shape() (inputs, outputs, weights int) {
	return b.n, b.n, b.n
}

func (b bias) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	w := mat.NewVecDense(b.n, h)
	dst.AddVec(x, w)
}

func (b bias) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, _ []float64) {
	dYdX.CloneFrom(b.identity)
	dYdH.CloneFrom(b.identity)
}
