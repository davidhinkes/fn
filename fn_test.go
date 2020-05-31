package fn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestVectorAssumptions(t *testing.T) {
	// See if the NewVecDense slice is really the underlying data
	// struct and mutations to either are equivlent.
	underlying := []float64{0}
	v := mat.NewVecDense(1, underlying)
	underlying[0] = 1
	if underlying[0] != v.AtVec(0) {
		t.Errorf("%v and %v should be the same", underlying, v)
	}
	v.SetVec(0, 2)
	if underlying[0] != v.AtVec(0) {
		t.Errorf("%v and %v should be the same", underlying, v)
	}
}
