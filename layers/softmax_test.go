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

func TestSoftmaxJacobian(t *testing.T) {
	n := 4
	s := softmax{n: n}
	x := mat.NewVecDense(n, []float64{0.5, -0.3, 1.2, 0.1})

	dYdX := mat.NewDense(n, n, nil)
	s.D(dYdX, nil, x, nil)

	// Verify against numerical differentiation.
	eps := 1e-7
	y := mat.NewVecDense(n, nil)
	yPerturbed := mat.NewVecDense(n, nil)
	s.F(y, x, nil)

	for j := 0; j < n; j++ {
		xp := mat.NewVecDense(n, nil)
		xp.CopyVec(x)
		xp.SetVec(j, xp.AtVec(j)+eps)
		s.F(yPerturbed, xp, nil)
		for i := 0; i < n; i++ {
			numerical := (yPerturbed.AtVec(i) - y.AtVec(i)) / eps
			analytical := dYdX.At(i, j)
			if math.Abs(numerical-analytical) > 1e-5 {
				t.Errorf("dY[%d]/dX[%d]: numerical=%v analytical=%v", i, j, numerical, analytical)
			}
		}
	}
}

func TestSoftmaxJacobianRowSum(t *testing.T) {
	n := 5
	s := softmax{n: n}
	x := mkRandomVec(n)

	dYdX := mat.NewDense(n, n, nil)
	s.D(dYdX, nil, x, nil)

	// Each row of the Jacobian must sum to 0, since the outputs
	// are constrained to sum to 1.
	for i := 0; i < n; i++ {
		var rowSum float64
		for j := 0; j < n; j++ {
			rowSum += dYdX.At(i, j)
		}
		if math.Abs(rowSum) > 1e-12 {
			t.Errorf("row %d sum=%v, want 0", i, rowSum)
		}
	}
}
