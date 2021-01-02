package layers

import (
	"fn"

	"math"
)

func sigmoid(x float64) float64 {
	return 1. / (math.Exp(-x) + 1.)
}

func dSigmoid(x float64) float64 {
	s := sigmoid(x)
	return s * (1. - s)
}

func MakeSigmoid() fn.Layer {
	return staticFunc{
		f: sigmoid,
		d: dSigmoid,
	}
}
