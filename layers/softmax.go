package layers

import (
	"math"

	"fn"
	"fn/matrixpool"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func Softmax() fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return wrapVectorLayer(softmax{n: inputs})
	}
}

type softmax struct {
	n int
}

// We expect softmax to adhere to the VectorLayer
var _ VectorLayer = softmax{}

func (s softmax) Shape() (inputs, outputs, weights int) {
	return s.n, s.n, 0
}

func (s softmax) F(dst *mat.VecDense, x mat.Vector, _ []float64) {
	raw := dst.RawVector().Data
	// Find max for numerical stability.
	m := x.AtVec(0)
	for i := 1; i < s.n; i++ {
		if v := x.AtVec(i); v > m {
			m = v
		}
	}
	var sum float64
	for i := range raw {
		e := math.Exp(x.AtVec(i) - m)
		raw[i] = e
		sum += e
	}
	floats.Scale(1/sum, raw)
}

func (s softmax) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64) {
	// Recompute forward output for the VJP.
	y := matrixpool.GetVec(s.n)
	defer matrixpool.PutVec(y)
	s.F(y, x, h)

	// VJP: dLdX[j] = y[j] * (dLdY[j] - dot(dLdY, y))
	yRaw := y.RawVector().Data
	dLdXRaw := dLdX.RawVector().Data
	// Read dLdY into dLdXRaw as scratch, then compute in-place.
	for i := range dLdXRaw {
		dLdXRaw[i] = dLdY.AtVec(i)
	}
	dot := floats.Dot(dLdXRaw, yRaw)
	for j := range dLdXRaw {
		dLdXRaw[j] = yRaw[j] * (dLdXRaw[j] - dot)
	}
}
