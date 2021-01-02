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
	ident := mat.NewDiagDense(n, nil)
	for i := 0; i < n; i++ {
		ident.SetDiag(i, 1.0)
	}
	b := bias{
		n:        n,
		identity: ident,
	}
	return b
}

func (b bias) NumHyperparameters() int {
	return b.n
}

func (b bias) F(x mat.Vector, h []float64) mat.Vector {
	w := mat.NewVecDense(b.n, h)
	var ret mat.VecDense
	ret.AddVec(x, w)
	return &ret
}

func (b bias) D(x mat.Vector, _ []float64) (mat.Matrix, mat.Matrix) {
	return b.identity, b.identity
}
