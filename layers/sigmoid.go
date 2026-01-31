package layers

import (
	"fn"

	"math"
)

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1. / (1. + math.Exp(-x))
	}
	// For x < 0, use exp(x)/(1 + exp(x)) to avoid overflow
	ex := math.Exp(x)
	return ex / (1. + ex)
}

func dSigmoid(x float64) float64 {
	if x >= 0 {
		ex := math.Exp(-x)
		denom := 1. + ex
		return ex / (denom * denom)
	}
	// For x < 0
	ex := math.Exp(x)
	denom := 1. + ex
	return ex / (denom * denom)
}

func Sigmoid() fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return staticFunc{
			f: sigmoid,
			d: dSigmoid,
			n: inputs,
		}
	}
}
