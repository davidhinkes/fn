package layers

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func randomize(v *mat.VecDense) {
	for i := 0; i < v.Len(); i++ {
		r := 2*rand.Float64() - 1
		v.SetVec(i, r)
	}
}

func diag(x mat.Vector) *mat.DiagDense {
	ret := mat.NewDiagDense(x.Len(), nil)
	for i := 0; i < x.Len(); i++ {
		ret.SetDiag(i, x.AtVec(i))
	}
	return ret
}
