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
	if a := diagFromSlice(d); !mat.Equal(a, m) {
		t.Errorf("%v and %v should be equal, but are not", mat.Formatted(a), mat.Formatted(m))
	}
	if a := diag(mat.NewVecDense(len(d), d)); !mat.Equal(a, m) {
		t.Errorf("%v and %v should be equal, but are not", mat.Formatted(a), mat.Formatted(m))
	}
}
