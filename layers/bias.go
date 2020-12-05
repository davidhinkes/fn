package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

type bias struct {
	w        *mat.VecDense
	identity mat.Matrix
}

func MakeBiasLayer(n int) fn.Layer {
	ident := mat.NewDiagDense(n, nil)
	for i := 0; i < n; i++ {
		ident.SetDiag(i, 1.0)
	}
	b := bias{
		w:        mat.NewVecDense(n, nil),
		identity: ident,
	}
	randomize(b.w)
	return b
}

func (b bias) Hyperparameters() *mat.VecDense {
	return b.w
}

func (b bias) F(x mat.Vector) mat.Vector {
	var ret mat.VecDense
	ret.AddVec(x, b.w)
	return &ret
}

func (b bias) D(x mat.Vector) (mat.Matrix, mat.Matrix) {
	return b.identity, b.identity
}
