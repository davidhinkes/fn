package fn

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

type Model struct {
	nodes []node
}

type node struct {
	layer           Layer
	weights []float64
}

func (m Model) Eval(x mat.Vector) (mat.Vector, []mat.Vector) {
	var upsilons []mat.Vector
	previousUpsilon := x
	for _, node := range m.nodes {
		u := node.layer.F(previousUpsilon, node.weights)
		upsilons = append(upsilons, u)
		previousUpsilon = u
	}
	return previousUpsilon, upsilons
}

// MakeModel will return a Model from layers.
func MakeModel(layers ...Layer) Model {
	model := Model{}
	for _, layer := range layers {
		model.nodes = append(model.nodes, node{
			layer:           layer,
			weights: random(layer.NumWeights()),
		})
	}
	return model
}

func random(n int) []float64 {
	ret := make([]float64, n)
	for i := range ret {
		ret[i] = 2*rand.Float64() - 1
	}
	return ret
}
