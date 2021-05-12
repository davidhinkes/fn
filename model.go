package fn

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

type Model struct {
	layer    Layer
	weights  []float64
	exampleX mat.Vector
	exampleY mat.Vector
}

func (m Model) Eval(x mat.Vector) mat.Vector {
	return m.layer.F(x, m.weights)
}

// MakeModel will return a Model from layers.
func MakeModel(layers ...Layer) Model {
	if len(layers) != 1 {
		return MakeModel(Serial(layers...))
	}
	layer := layers[0]
	return Model{
		layer:   layer,
		weights: random(layer.NumWeights()),
	}
}

func random(n int) []float64 {
	ret := make([]float64, n)
	for i := range ret {
		ret[i] = 2*rand.Float64() - 1
	}
	return ret
}
