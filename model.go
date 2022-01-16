package fn

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
	yaml "gopkg.in/yaml.v2"
	"math/rand"
)

type Model struct {
	layer   Layer
	weights []float64
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
	Weights  []float64
	ExampleX []float64
	ExampleY []float64
}

// Marshal marshals the Model. Variable x is an example input vector
// that is used to make a checksum-like mechanism to verify the integrity
// of a Model.
func (m *Model) Marshal(x mat.Vector) ([]byte, error) {
	s := storage{
		Weights: m.weights,
	}
	if x != nil {
		s.ExampleX = toSlice(x)
		s.ExampleY = toSlice(m.Eval(x))
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
	a, b := m.weights, s.Weights
	if len(a) != len(b) {
		log.Fatalf("Unmarshal: weights cardinality missmatch")
	}
	copy(a, b)
	if len(s.ExampleX) == 0 {
		return nil
	}
	return m.testExample(toVec(s.ExampleX), toVec(s.ExampleY))
}

func (m *Model) testExample(x, y mat.Vector) error {
	var e mat.VecDense
	e.SubVec(y, m.Eval(x))
	if s := mat.Dot(&e, &e); s != 0 {
		return fmt.Errorf("examples do not match; The Model is not compatible with the supplied weights. (squared error=%v)", s)
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
