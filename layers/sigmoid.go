package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct{}

func (_ Sigmoid) F(x mat.Vector) mat.Vector {
	n := x.Len()
	s := make([]float64, n)
	for i, _ := range s {
		s[i] = 1. / (math.Exp(-x.AtVec(i)) + 1.)
	}
	return mat.NewVecDense(n, s)
}

func (sig Sigmoid) D(x mat.Vector) (mat.Matrix, mat.Matrix) {
	y := sig.F(x)
	n := x.Len()
	m := mat.NewDiagDense(n, nil)
	for i := 0; i < n; i++ {
		f := y.AtVec(i)
		m.SetDiag(i, f*(1.-f))
	}
	return m, nil
}

func (_ Sigmoid) Hyperparameters() *mat.VecDense {
	return nil
}
