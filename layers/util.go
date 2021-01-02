package layers

import (
	"gonum.org/v1/gonum/mat"
)

func diag(x mat.Vector) *mat.DiagDense {
	ret := mat.NewDiagDense(x.Len(), nil)
	for i := 0; i < x.Len(); i++ {
		ret.SetDiag(i, x.AtVec(i))
	}
	return ret
}
