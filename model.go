package fn

import (
	"log"
	"errors"

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
	Weights []float64
	ExampleX []float64
	ExampleY []float64
}

func (m *Model) Marshal() ([]byte,error) {
	s := storage {
		Weights: m.weights,
		ExampleX: toSlice(m.exampleX),
		ExampleY: toSlice(m.exampleY),
	}
	return yaml.Marshal(s)
}

func toSlice(x mat.Vector) []float64 {
	if x == nil {
		return nil
	}
	ret := make([]float64, x.Len())
	for i := range ret {
		ret[i] = x.AtVec(i)
	}
	return ret
}

func toVec(x []float64) mat.Vector {
	if len(x) == 0 {
		return nil
	}
	return mat.NewVecDense(len(x), x)
}

func (m *Model) Unmarshal(bytes []byte) error {
	var s storage
	if err := yaml.Unmarshal(bytes, &s); err != nil {
		return err
	}
	a,b := m.weights, s.Weights
	if len(a) != len(b) {log.Fatalf("Unmarshal: weights cardinality missmatch")}
	copy(a,b)
	m.exampleX = toVec(s.ExampleX)
	m.exampleY = toVec(s.ExampleY)
	return m.testExample()
}

func (m *Model) testExample() error {
	if m.exampleX == nil {
		// nothing to test
		return nil
	}
	y := m.Eval(m.exampleX)
	var e mat.VecDense
	e.SubVec(y,m.exampleY)
	if mat.Dot(&e,&e) != 0 {
		return errors.New("examples do not match. The Model is not compatible with the supplied weights.")
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
