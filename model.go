package fn

import (
	"log"

	"gonum.org/v1/gonum/mat"
	"math/rand"
	yaml "gopkg.in/yaml.v2"
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

type storage struct {
	Weights [][]float64
}

func (m Model) Marshal() ([]byte,error) {
	var s storage 
	for _,node := range m.nodes {
		s.Weights = append(s.Weights, node.weights)
	}
	return yaml.Marshal(s)
}

func (m Model) Unmarshal(bytes []byte) error {
	var s storage
	if err := yaml.Unmarshal(bytes, &s); err != nil {
		return err
	}
	for i,node := range m.nodes {
		a := node.weights
		b := s.Weights[i]
		if len(a) != len(b) {log.Fatalf("Unmarshal: weights cardinality missmatch")}
		copy(a,b)
	}
	return nil
}

func random(n int) []float64 {
	ret := make([]float64, n)
	for i := range ret {
		ret[i] = 2*rand.Float64() - 1
	}
	return ret
}
