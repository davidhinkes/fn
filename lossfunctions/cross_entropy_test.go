package lossfunctions

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCrossEntropyOneHot(t *testing.T) {
	ce := NewCrossEntropy()
	// Prediction is a probability distribution; target is one-hot with class 1.
	y := mat.NewVecDense(3, []float64{0.2, 0.7, 0.1})
	yHat := mat.NewVecDense(3, []float64{0, 1, 0})

	grad := mat.NewVecDense(3, nil)
	loss := ce.F(grad, y, yHat)
	want := -math.Log(0.7)
	if math.Abs(loss-want) > 1e-12 {
		t.Errorf("loss=%v, want %v", loss, want)
	}
}

func TestCrossEntropyPerfectPrediction(t *testing.T) {
	ce := NewCrossEntropy()
	y := mat.NewVecDense(3, []float64{0, 1, 0})
	yHat := mat.NewVecDense(3, []float64{0, 1, 0})

	grad := mat.NewVecDense(3, nil)
	loss := ce.F(grad, y, yHat)
	// -1 * log(1) = 0
	if math.Abs(loss) > 1e-12 {
		t.Errorf("loss=%v, want 0", loss)
	}
}

func TestCrossEntropyNearZeroPrediction(t *testing.T) {
	ce := NewCrossEntropy()
	// Prediction assigns near-zero probability to the correct class.
	y := mat.NewVecDense(3, []float64{0.99, 0.005, 0.005})
	yHat := mat.NewVecDense(3, []float64{0, 1, 0})

	grad := mat.NewVecDense(3, nil)
	loss := ce.F(grad, y, yHat)
	if loss < 1.0 {
		t.Errorf("loss=%v, want large value for wrong prediction", loss)
	}
}

func TestCrossEntropyZeroPredictionClamped(t *testing.T) {
	ce := NewCrossEntropy()
	// Exact zero in prediction should not produce Inf or NaN.
	y := mat.NewVecDense(3, []float64{0, 0, 1})
	yHat := mat.NewVecDense(3, []float64{1, 0, 0})

	grad := mat.NewVecDense(3, nil)
	loss := ce.F(grad, y, yHat)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Errorf("loss=%v, want finite", loss)
	}
	for i := 0; i < grad.Len(); i++ {
		if math.IsNaN(grad.AtVec(i)) || math.IsInf(grad.AtVec(i), 0) {
			t.Errorf("grad[%d]=%v, want finite", i, grad.AtVec(i))
		}
	}
}

func TestCrossEntropyGradient(t *testing.T) {
	ce := NewCrossEntropy()
	y := mat.NewVecDense(3, []float64{0.2, 0.7, 0.1})
	yHat := mat.NewVecDense(3, []float64{0, 1, 0})

	grad := mat.NewVecDense(3, nil)
	ce.F(grad, y, yHat)

	// Verify against numerical differentiation.
	eps := 1e-7
	for i := 0; i < y.Len(); i++ {
		yp := mat.NewVecDense(y.Len(), nil)
		yp.CopyVec(y)
		yp.SetVec(i, yp.AtVec(i)+eps)
		tmp := mat.NewVecDense(3, nil)
		lossPlus := ce.F(tmp, yp, yHat)

		ym := mat.NewVecDense(y.Len(), nil)
		ym.CopyVec(y)
		ym.SetVec(i, ym.AtVec(i)-eps)
		lossMinus := ce.F(tmp, ym, yHat)

		numerical := (lossPlus - lossMinus) / (2 * eps)
		analytical := grad.AtVec(i)
		if math.Abs(numerical-analytical) > 1e-5 {
			t.Errorf("grad[%d]: numerical=%v analytical=%v", i, numerical, analytical)
		}
	}
}

func TestCrossEntropyGradientZeroTarget(t *testing.T) {
	ce := NewCrossEntropy()
	y := mat.NewVecDense(3, []float64{0.2, 0.7, 0.1})
	yHat := mat.NewVecDense(3, []float64{0, 1, 0})

	grad := mat.NewVecDense(3, nil)
	ce.F(grad, y, yHat)

	// Gradient should be zero for classes with zero target.
	if grad.AtVec(0) != 0 {
		t.Errorf("grad[0]=%v, want 0 for zero target", grad.AtVec(0))
	}
	if grad.AtVec(2) != 0 {
		t.Errorf("grad[2]=%v, want 0 for zero target", grad.AtVec(2))
	}

	// Gradient for the correct class should be negative (push probability up).
	if grad.AtVec(1) >= 0 {
		t.Errorf("grad[1]=%v, want negative", grad.AtVec(1))
	}
}
