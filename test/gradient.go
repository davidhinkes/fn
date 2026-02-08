package test

import (
	"math"
	"math/rand"
	"testing"

	"fn"

	"gonum.org/v1/gonum/mat"
)

// GradientCheck verifies that a layer's analytical gradients (from D) match
// numerical gradients computed via finite differences. Fails the test if the
// relative error exceeds 1e-5.
func GradientCheck(t *testing.T, layer fn.Layer) {
	t.Helper()
	GradientCheckWithSeed(t, layer, 42)
}

// GradientCheckWithSeed is like GradientCheck but accepts a random seed for reproducibility.
func GradientCheckWithSeed(t *testing.T, layer fn.Layer, seed int64) {
	t.Helper()

	const eps = 1e-5
	const threshold = 1e-5

	rng := rand.New(rand.NewSource(seed))
	inputs, outputs, weights := layer.Shape()

	// Generate random inputs, weights, and target
	x := randSlice(rng, inputs)
	h := randSlice(rng, weights)
	target := randSlice(rng, outputs)

	dLdXErr, dLdHErr := computeGradientErrors(layer, x, h, target, eps)

	if dLdXErr > threshold {
		t.Errorf("dLdX gradient error too high: %e > %e", dLdXErr, threshold)
	}
	if dLdHErr > threshold {
		t.Errorf("dLdH gradient error too high: %e > %e", dLdHErr, threshold)
	}
}

func computeGradientErrors(layer fn.Layer, x, h, target []float64, eps float64) (dLdXErr, dLdHErr float64) {
	inputs, outputs, weights := layer.Shape()

	// Wrap into 1-row Dense matrices (batch size 1).
	X := mat.NewDense(1, inputs, x)
	targetVec := mat.NewVecDense(outputs, target)

	// Allocate buffers
	Y := mat.NewDense(1, outputs, nil)
	dLdY := mat.NewDense(1, outputs, nil)
	dLdX := mat.NewDense(1, inputs, nil)
	var dLdH *mat.VecDense
	if weights > 0 {
		dLdH = mat.NewVecDense(weights, nil)
	}

	// Forward pass and compute dLdY (using squared error loss)
	layer.F(Y, X, h)
	yVec := mat.NewVecDense(outputs, nil)
	for i := 0; i < outputs; i++ {
		yVec.SetVec(i, Y.At(0, i))
	}
	yVec.SubVec(yVec, targetVec) // dL/dY = y - target for L = 0.5*||y-target||^2
	for i := 0; i < outputs; i++ {
		dLdY.Set(0, i, yVec.AtVec(i))
	}

	// Analytical gradients
	layer.D(dLdX, dLdH, dLdY, X, h)

	// Numerical gradient for dLdX
	YPlus := mat.NewDense(1, outputs, nil)
	YMinus := mat.NewDense(1, outputs, nil)
	xPerturbed := make([]float64, inputs)
	copy(xPerturbed, x)
	XPerturbed := mat.NewDense(1, inputs, xPerturbed)

	for i := 0; i < inputs; i++ {
		orig := xPerturbed[i]

		xPerturbed[i] = orig + eps
		layer.F(YPlus, XPerturbed, h)
		lossPlus := squaredErrorDense(YPlus, targetVec)

		xPerturbed[i] = orig - eps
		layer.F(YMinus, XPerturbed, h)
		lossMinus := squaredErrorDense(YMinus, targetVec)

		xPerturbed[i] = orig

		numerical := (lossPlus - lossMinus) / (2 * eps)
		analytical := dLdX.At(0, i)
		dLdXErr = max(dLdXErr, relativeError(analytical, numerical))
	}

	// Numerical gradient for dLdH
	if weights > 0 {
		hPerturbed := make([]float64, len(h))
		copy(hPerturbed, h)

		for i := 0; i < weights; i++ {
			orig := hPerturbed[i]

			hPerturbed[i] = orig + eps
			layer.F(YPlus, X, hPerturbed)
			lossPlus := squaredErrorDense(YPlus, targetVec)

			hPerturbed[i] = orig - eps
			layer.F(YMinus, X, hPerturbed)
			lossMinus := squaredErrorDense(YMinus, targetVec)

			hPerturbed[i] = orig

			numerical := (lossPlus - lossMinus) / (2 * eps)
			analytical := dLdH.AtVec(i)
			dLdHErr = max(dLdHErr, relativeError(analytical, numerical))
		}
	}

	return dLdXErr, dLdHErr
}

// squaredErrorDense computes 0.5*||Y[0,:] - target||^2 for a 1-row Dense.
func squaredErrorDense(Y *mat.Dense, target *mat.VecDense) float64 {
	_, cols := Y.Dims()
	var sum float64
	for i := 0; i < cols; i++ {
		d := Y.At(0, i) - target.AtVec(i)
		sum += d * d
	}
	return 0.5 * sum
}

func relativeError(analytical, numerical float64) float64 {
	denom := math.Max(math.Abs(analytical), math.Abs(numerical))
	if denom < 1e-10 {
		return 0 // both essentially zero
	}
	return math.Abs(analytical-numerical) / denom
}

func randSlice(rng *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = rng.NormFloat64()
	}
	return s
}
