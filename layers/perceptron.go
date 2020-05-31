package layers

import (
	"fn"

	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type ActivationFunction interface {
	F(float64) (float64, float64)
}

type Sigmoid struct{}

func (s Sigmoid) F(x float64) (float64, float64) {
	f := 1. / (math.Exp(-x) + 1.)
	d := f * (1 - f)
	return f, d
}

type Identity struct{}

func (i Identity) F(x float64) (float64, float64) {
	return x, 1.0
}

func makeHyperparameters(inputs, outputs int) (*mat.Dense, *mat.VecDense, *mat.VecDense) {
	s := make([]float64, inputs*outputs+outputs)
	n := inputs * outputs
	return mat.NewDense(outputs, inputs, s[:n]), mat.NewVecDense(outputs, s[n:]), mat.NewVecDense(len(s), s)
}

func MakePerceptronLayer(inputs, outputs int, a ActivationFunction) fn.Layer {
	w, b, h := makeHyperparameters(inputs, outputs)
	randomize(h)
	return &perceptron{
		hyperparamaters:    h,
		w:                  w,
		b:                  b,
		activationFunction: a,
	}
}

func randomize(v *mat.VecDense) {
	for i := 0; i < v.Len(); i++ {
		r := 2*rand.Float64() - 1
		v.SetVec(i, r)
	}
}

func random(n int) []float64 {
	data := make([]float64, n)
	for i := range data {
		data[i] = 2*rand.Float64() - 1
	}
	return data
}

type perceptron struct {
	hyperparamaters    *mat.VecDense
	w                  *mat.Dense
	b                  *mat.VecDense
	activationFunction ActivationFunction
}

func (s *perceptron) Learn(v mat.Vector) {
	s.hyperparamaters.AddVec(s.hyperparamaters, v)
}

func (s *perceptron) F(x mat.Vector) mat.Vector {
	r, _ := s.w.Dims()
	o := mat.NewVecDense(r, nil)
	o.MulVec(s.w, x)
	o.AddVec(o, s.b)
	for i := 0; i < o.Len(); i++ {
		f, _ := s.activationFunction.F(o.AtVec(i))
		o.SetVec(i, f)
	}
	return o
}

func (s *perceptron) Backpropagate(x mat.Vector, dLoss mat.Vector) (mat.Vector, mat.Vector) {
	outputs, inputs := s.w.Dims()
	dLossDW, dLossDB, ret := makeHyperparameters(inputs, outputs)
	// update calculate dLossDB
	for i := 0; i < dLoss.Len(); i++ {
		_, d := s.activationFunction.F(mat.Dot(s.w.RowView(i), x) + s.b.AtVec(i))
		v := dLossDB.AtVec(i) + dLoss.AtVec(i)*d
		dLossDB.SetVec(i, v)
	}

	// calculate dLossDW
	dLossDW.Apply(func(i, j int, v float64) float64 {
		_, d := s.activationFunction.F(mat.Dot(s.w.RowView(i), x) + s.b.AtVec(i))
		return v + dLoss.AtVec(i)*d*x.AtVec(j)
	}, dLossDW)

	// Calculate dLossDX
	dLossDX := mat.NewVecDense(x.Len(), nil)
	for j := 0; j < x.Len(); j++ {
		var sum float64
		for i := 0; i < dLoss.Len(); i++ {
			_, d := s.activationFunction.F(mat.Dot(s.w.RowView(i), x) + s.b.AtVec(i))
			sum += dLoss.AtVec(i) * d * s.w.At(i, j)
		}
		dLossDX.SetVec(j, sum)
	}
	return dLossDX, ret
}
