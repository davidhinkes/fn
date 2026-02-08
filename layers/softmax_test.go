package layers

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmaxShape(t *testing.T) {
	s := softmax{n: 5}
	inputs, outputs, weights := s.Shape()
	if inputs != 5 || outputs != 5 || weights != 0 {
		t.Errorf("got Shape()=(%d,%d,%d), want (5,5,0)", inputs, outputs, weights)
	}
}

func TestSoftmaxForward(t *testing.T) {
	s := softmax{n: 4}
	X := mat.NewDense(1, 4, []float64{1, 2, 3, 4})
	dst := mat.NewDense(1, 4, nil)
	s.F(dst, X, nil)

	// All outputs must be positive.
	for i := 0; i < 4; i++ {
		if v := dst.At(0, i); v <= 0 {
			t.Errorf("output[%d]=%v, want > 0", i, v)
		}
	}

	// Outputs must sum to 1.
	var sum float64
	for i := 0; i < 4; i++ {
		sum += dst.At(0, i)
	}
	if math.Abs(sum-1.0) > 1e-12 {
		t.Errorf("sum=%v, want 1.0", sum)
	}

	// Larger inputs must produce larger outputs.
	for i := 0; i < 3; i++ {
		if dst.At(0, i) >= dst.At(0, i+1) {
			t.Errorf("output[%d]=%v >= output[%d]=%v, want strictly increasing",
				i, dst.At(0, i), i+1, dst.At(0, i+1))
		}
	}
}

func TestSoftmaxUniformInput(t *testing.T) {
	n := 5
	s := softmax{n: n}
	X := mat.NewDense(1, n, []float64{3, 3, 3, 3, 3})
	dst := mat.NewDense(1, n, nil)
	s.F(dst, X, nil)

	want := 1.0 / float64(n)
	for i := 0; i < n; i++ {
		if math.Abs(dst.At(0, i)-want) > 1e-12 {
			t.Errorf("output[%d]=%v, want %v", i, dst.At(0, i), want)
		}
	}
}

func TestSoftmaxNumericalStability(t *testing.T) {
	s := softmax{n: 3}
	X := mat.NewDense(1, 3, []float64{1000, 1001, 1002})
	dst := mat.NewDense(1, 3, nil)
	s.F(dst, X, nil)

	for i := 0; i < 3; i++ {
		v := dst.At(0, i)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("output[%d]=%v, want finite", i, v)
		}
	}

	var sum float64
	for i := 0; i < 3; i++ {
		sum += dst.At(0, i)
	}
	if math.Abs(sum-1.0) > 1e-12 {
		t.Errorf("sum=%v, want 1.0", sum)
	}
}

func TestSoftmaxVJP(t *testing.T) {
	n := 4
	s := softmax{n: n}
	X := mat.NewDense(1, n, []float64{0.5, -0.3, 1.2, 0.1})
	dLdY := mat.NewDense(1, n, []float64{0.1, -0.2, 0.3, -0.1})

	dLdX := mat.NewDense(1, n, nil)
	s.D(dLdX, nil, dLdY, X, nil)

	// Verify against numerical differentiation.
	eps := 1e-7
	Y := mat.NewDense(1, n, nil)
	s.F(Y, X, nil)

	// L = dot(dLdY[0,:], Y[0,:])
	var L float64
	for i := 0; i < n; i++ {
		L += dLdY.At(0, i) * Y.At(0, i)
	}

	yPerturbed := mat.NewDense(1, n, nil)
	for j := 0; j < n; j++ {
		xp := mat.NewDense(1, n, nil)
		xp.Copy(X)
		xp.Set(0, j, xp.At(0, j)+eps)
		s.F(yPerturbed, xp, nil)
		var Lp float64
		for i := 0; i < n; i++ {
			Lp += dLdY.At(0, i) * yPerturbed.At(0, i)
		}
		numerical := (Lp - L) / eps
		analytical := dLdX.At(0, j)
		if math.Abs(numerical-analytical) > 1e-5 {
			t.Errorf("dL/dX[%d]: numerical=%v analytical=%v", j, numerical, analytical)
		}
	}
}

func TestSoftmaxVJPUniformUpstream(t *testing.T) {
	n := 5
	s := softmax{n: n}
	X := mat.NewDense(1, n, mkRandomSlice(n))

	// If dLdY is uniform (all ones), dLdX should be zero.
	dLdYData := make([]float64, n)
	for i := range dLdYData {
		dLdYData[i] = 1.0
	}
	dLdY := mat.NewDense(1, n, dLdYData)
	dLdX := mat.NewDense(1, n, nil)
	s.D(dLdX, nil, dLdY, X, nil)

	for j := 0; j < n; j++ {
		if math.Abs(dLdX.At(0, j)) > 1e-12 {
			t.Errorf("dLdX[%d]=%v, want 0", j, dLdX.At(0, j))
		}
	}
}
