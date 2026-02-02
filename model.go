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

func (m Model) Eval(dst *mat.VecDense, x mat.Vector) {
	m.layer.F(dst, x, m.weights)
}

// MakeModel will return a Model from LayerBuilders.
func MakeModel(inputs int, builders ...LayerBuilder) Model {
	// combine into a single layer
	layer := SerialBuilder(builders...)(inputs)
	_, _, numWeights := layer.Shape()
	return Model{
		layer:   layer,
		weights: random(numWeights),
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
		_, outputs, _ := m.layer.Shape()
		y := mat.NewVecDense(outputs, nil)
		m.Eval(y, x)
		s.ExampleX = toSlice(x)
		s.ExampleY = toSlice(y)
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
		log.Fatalf("Unmarshal: weights cardinality mismatch")
	}
	copy(a, b)
	if len(s.ExampleX) == 0 {
		return nil
	}
	return m.testExample(toVec(s.ExampleX), toVec(s.ExampleY))
}

func (m *Model) testExample(x, y mat.Vector) error {
	_, outputs, _ := m.layer.Shape()
	yPrime := mat.NewVecDense(outputs, nil)
	m.Eval(yPrime, x)
	var e mat.VecDense
	e.SubVec(y, yPrime)
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
