package layers

import (
	"fn"
	"math"
)

func MakeRelu() fn.Layer {
	return staticFunc{
		f: func(x float64) float64 {
			return math.Max(.1*x, x)
		},
		d: func(x float64) float64 {
			if x < 0 {
				return .1
			}
			return 1.
		},
	}
}
