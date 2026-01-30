package layers

import (
	"testing"

	"fn"
	"gonum.org/v1/gonum/mat"
)

func TestParallel(t *testing.T) {
	p := fn.Parallel(MakeScalarLayer(10), MakeScalarLayer(10), MakeScalarLayer(10))
	if got, want := p.NumWeights(), 30; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	h := make([]float64, 30)
	x := mat.NewVecDense(10, nil)
	var y mat.VecDense
	p.F(&y, x, h)
	if y.Len() != 30 {
		t.Errorf("got %v, want 30", y.Len())
	}
	var dydx, dydh mat.Dense
	p.D(&dydx, &dydh, x, h)
	if r, c := dydx.Dims(); r != 30 || c != 10 {
		t.Errorf("got %vx%v; want 30x10", r, c)
	}
	if r, c := dydh.Dims(); r != 30 || c != 30 {
		t.Errorf("got %vx%v; want 30x30", r, c)
	}
}
