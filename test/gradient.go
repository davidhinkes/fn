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
	x := randVec(rng, inputs)
	h := randSlice(rng, weights)
	target := randVec(rng, outputs)

	dLdXErr, dLdHErr := computeGradientErrors(layer, x, h, target, eps)

	if dLdXErr > threshold {
		t.Errorf("dLdX gradient error too high: %e > %e", dLdXErr, threshold)
	}
	if dLdHErr > threshold {
		t.Errorf("dLdH gradient error too high: %e > %e", dLdHErr, threshold)
	}
}

func computeGradientErrors(layer fn.Layer, x *mat.VecDense, h []float64, target *mat.VecDense, eps float64) (dLdXErr, dLdHErr float64) {
	inputs, outputs, weights := layer.Shape()

	// Allocate buffers
	y := mat.NewVecDense(outputs, nil)
	dLdY := mat.NewVecDense(outputs, nil)
	dLdX := mat.NewVecDense(inputs, nil)
	var dLdH *mat.VecDense
	if weights > 0 {
		dLdH = mat.NewVecDense(weights, nil)
	}

	// Forward pass and compute dLdY (using squared error loss)
	layer.F(y, x, h)
	dLdY.SubVec(y, target) // dL/dY = y - target for L = 0.5*||y-target||^2

	// Analytical gradients
	layer.D(dLdX, dLdH, dLdY, x, h)

	// Numerical gradient for dLdX
	yPlus := mat.NewVecDense(outputs, nil)
	yMinus := mat.NewVecDense(outputs, nil)
	xPerturbed := mat.NewVecDense(inputs, nil)
	xPerturbed.CopyVec(x)

	for i := 0; i < inputs; i++ {
		orig := xPerturbed.AtVec(i)

		xPerturbed.SetVec(i, orig+eps)
		layer.F(yPlus, xPerturbed, h)
		lossPlus := squaredError(yPlus, target)

		xPerturbed.SetVec(i, orig-eps)
		layer.F(yMinus, xPerturbed, h)
		lossMinus := squaredError(yMinus, target)

		xPerturbed.SetVec(i, orig)

		numerical := (lossPlus - lossMinus) / (2 * eps)
		analytical := dLdX.AtVec(i)
		dLdXErr = max(dLdXErr, relativeError(analytical, numerical))
	}

	// Numerical gradient for dLdH
	if weights > 0 {
		hPerturbed := make([]float64, len(h))
		copy(hPerturbed, h)

		for i := 0; i < weights; i++ {
			orig := hPerturbed[i]

			hPerturbed[i] = orig + eps
			layer.F(yPlus, x, hPerturbed)
			lossPlus := squaredError(yPlus, target)

			hPerturbed[i] = orig - eps
			layer.F(yMinus, x, hPerturbed)
			lossMinus := squaredError(yMinus, target)

			hPerturbed[i] = orig

			numerical := (lossPlus - lossMinus) / (2 * eps)
			analytical := dLdH.AtVec(i)
			dLdHErr = max(dLdHErr, relativeError(analytical, numerical))
		}
	}

	return dLdXErr, dLdHErr
}

func squaredError(y, target *mat.VecDense) float64 {
	diff := mat.NewVecDense(y.Len(), nil)
	diff.SubVec(y, target)
	return 0.5 * mat.Dot(diff, diff)
}

func relativeError(analytical, numerical float64) float64 {
	denom := math.Max(math.Abs(analytical), math.Abs(numerical))
	if denom < 1e-10 {
		return 0 // both essentially zero
	}
	return math.Abs(analytical-numerical) / denom
}

func randVec(rng *rand.Rand, n int) *mat.VecDense {
	return mat.NewVecDense(n, randSlice(rng, n))
}

func randSlice(rng *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = rng.NormFloat64()
	}
	return s
}
