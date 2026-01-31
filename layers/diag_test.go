package layers

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestDiag(t *testing.T) {
	d := []float64{1, 2, 3}
	m := mat.NewDense(len(d), len(d), nil)
	for i, v := range d {
		m.Set(i, i, v)
	}
	a := mat.NewDense(len(d), len(d), nil)
	diagFromSlice(a, d)
	if !mat.Equal(a, m) {
		t.Errorf("%v and %v should be equal, but are not", mat.Formatted(a), mat.Formatted(m))
	}
	b := mat.NewDense(len(d), len(d), nil)
	diag(b, mat.NewVecDense(len(d), d))
	if !mat.Equal(b, m) {
		t.Errorf("%v and %v should be equal, but are not", mat.Formatted(b), mat.Formatted(m))
	}
}
