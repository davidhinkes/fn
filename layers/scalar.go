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

func (s scalar) F(x mat.Vector, h []float64) mat.Vector {
	var ret mat.VecDense
	ret.MulElemVec(x, mat.NewVecDense(s.n, h))
	return &ret
}

func (s scalar) D(x mat.Vector, h []float64) (mat.Matrix, mat.Matrix) {
	return mat.NewDiagDense(len(h), h), diag(x)
}
