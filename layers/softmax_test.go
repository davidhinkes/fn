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
	x := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	dst := mat.NewVecDense(4, nil)
	s.F(dst, x, nil)

	// All outputs must be positive.
	for i := 0; i < 4; i++ {
		if v := dst.AtVec(i); v <= 0 {
			t.Errorf("output[%d]=%v, want > 0", i, v)
		}
	}

	// Outputs must sum to 1.
	var sum float64
	for i := 0; i < 4; i++ {
		sum += dst.AtVec(i)
	}
	if math.Abs(sum-1.0) > 1e-12 {
		t.Errorf("sum=%v, want 1.0", sum)
	}

	// Larger inputs must produce larger outputs.
	for i := 0; i < 3; i++ {
		if dst.AtVec(i) >= dst.AtVec(i+1) {
			t.Errorf("output[%d]=%v >= output[%d]=%v, want strictly increasing",
				i, dst.AtVec(i), i+1, dst.AtVec(i+1))
		}
	}
}

func TestSoftmaxUniformInput(t *testing.T) {
	n := 5
	s := softmax{n: n}
	x := mat.NewVecDense(n, []float64{3, 3, 3, 3, 3})
	dst := mat.NewVecDense(n, nil)
	s.F(dst, x, nil)

	want := 1.0 / float64(n)
	for i := 0; i < n; i++ {
		if math.Abs(dst.AtVec(i)-want) > 1e-12 {
			t.Errorf("output[%d]=%v, want %v", i, dst.AtVec(i), want)
		}
	}
}

func TestSoftmaxNumericalStability(t *testing.T) {
	s := softmax{n: 3}
	x := mat.NewVecDense(3, []float64{1000, 1001, 1002})
	dst := mat.NewVecDense(3, nil)
	s.F(dst, x, nil)

	for i := 0; i < 3; i++ {
		v := dst.AtVec(i)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("output[%d]=%v, want finite", i, v)
		}
	}

	var sum float64
	for i := 0; i < 3; i++ {
		sum += dst.AtVec(i)
	}
	if math.Abs(sum-1.0) > 1e-12 {
		t.Errorf("sum=%v, want 1.0", sum)
	}
}

func TestSoftmaxVJP(t *testing.T) {
	n := 4
	s := softmax{n: n}
	x := mat.NewVecDense(n, []float64{0.5, -0.3, 1.2, 0.1})
	dLdY := mat.NewVecDense(n, []float64{0.1, -0.2, 0.3, -0.1})

	dLdX := mat.NewVecDense(n, nil)
	s.D(dLdX, nil, dLdY, x, nil)

	// Verify against numerical differentiation.
	eps := 1e-7
	y := mat.NewVecDense(n, nil)
	s.F(y, x, nil)

	// L = dot(dLdY, y)
	var L float64
	for i := 0; i < n; i++ {
		L += dLdY.AtVec(i) * y.AtVec(i)
	}

	yPerturbed := mat.NewVecDense(n, nil)
	for j := 0; j < n; j++ {
		xp := mat.NewVecDense(n, nil)
		xp.CopyVec(x)
		xp.SetVec(j, xp.AtVec(j)+eps)
		s.F(yPerturbed, xp, nil)
		var Lp float64
		for i := 0; i < n; i++ {
			Lp += dLdY.AtVec(i) * yPerturbed.AtVec(i)
		}
		numerical := (Lp - L) / eps
		analytical := dLdX.AtVec(j)
		if math.Abs(numerical-analytical) > 1e-5 {
			t.Errorf("dL/dX[%d]: numerical=%v analytical=%v", j, numerical, analytical)
		}
	}
}

func TestSoftmaxVJPUniformUpstream(t *testing.T) {
	n := 5
	s := softmax{n: n}
	x := mat.NewVecDense(n, mkRandomSlice(n))

	// If dLdY is uniform (all ones), dLdX should be zero.
	dLdYData := make([]float64, n)
	for i := range dLdYData {
		dLdYData[i] = 1.0
	}
	dLdY := mat.NewVecDense(n, dLdYData)
	dLdX := mat.NewVecDense(n, nil)
	s.D(dLdX, nil, dLdY, x, nil)

	for j := 0; j < n; j++ {
		if math.Abs(dLdX.AtVec(j)) > 1e-12 {
			t.Errorf("dLdX[%d]=%v, want 0", j, dLdX.AtVec(j))
		}
	}
}
