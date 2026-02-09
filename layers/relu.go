package layers

import (
	"math"

	"fn"
)

func Relu() fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return wrapVectorLayer(staticFunc{
			f: func(x float64) float64 {
				return math.Max(.1*x, x)
			},
			d: func(x float64) float64 {
				if x < 0 {
					return .1
				}
				return 1.
			},
			n: inputs,
		})
	}
}
