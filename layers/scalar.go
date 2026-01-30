package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

type scalar struct {
	n int
}

func MakeScalarLayer(n int) fn.Layer {
	return scalar{
		n: n,
	}
}

func (s scalar) NumWeights() int {
	return s.n
}

func (s scalar) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	dst.MulElemVec(x, mat.NewVecDense(s.n, h))
}

func (s scalar) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64) {
	dYdX.CloneFrom(diagFromSlice(h))
	if dYdH != nil {
		dYdH.CloneFrom(diag(x))
	}
}
