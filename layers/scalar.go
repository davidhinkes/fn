package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

type scalar struct {
	w *mat.VecDense
}

func MakeScalarLayer(n int) fn.Layer {
	s := scalar{
		w: mat.NewVecDense(n, nil),
	}
	randomize(s.w)
	return s
}

func (s scalar) Hyperparameters() *mat.VecDense {
	return s.w
}

func (s scalar) F(x mat.Vector) mat.Vector {
	var ret mat.VecDense
	ret.MulElemVec(x, s.w)
	return &ret
}

func (s scalar) D(x mat.Vector) (mat.Matrix, mat.Matrix) {
	return diag(s.w), diag(x)
}
