package layers

import (
	"fn"
	ma "fn/internal"

	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type ActivationFunction interface {
	F(float64)float64
	D(float64)float64
}

type Sigmoid struct {}

func (s Sigmoid) F(x float64)float64 {
	return 1. / (math.Exp(-x) + 1.)
}
func (s Sigmoid) D(x float64)float64 {
	v := s.F(x)
	return v*(1-v)
}

type Identity struct {}

func (i Identity) F(x float64)float64 {
	return x
}

func (i Identity) D(x float64)float64 {
	return 1.0
}

func MakePerceptronLayer(inputs, outputs int, a ActivationFunction) fn.Layer {
	return &perceptron{
		w: mat.NewDense(outputs, inputs, random(inputs*outputs)),
		b: mat.NewVecDense(outputs, random(outputs)),
		activationFunction: a,
		matArena: ma.Make(),
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
	w *mat.Dense
	b *mat.VecDense
	activationFunction ActivationFunction
	matArena ma.T
}

func (s *perceptron) F(x mat.Vector) mat.Vector {
	s.matArena.Reset()
	r,_ := s.w.Dims()
	o, _ := s.matArena.NewVecDense(r)
	o.MulVec(s.w, x)
	o.AddVec(o, s.b)
	for i := 0; i < o.Len(); i++ {
		v := o.AtVec(i)
		o.SetVec(i, s.activationFunction.F(v))
	}
	return o
}

func (s* perceptron) Learn(x mat.Vector, dLoss mat.Vector, alpha float64) mat.Vector {
	
	outputs,inputs := s.w.Dims()

	// calculate dLossDB
	dLossDB, _ := s.matArena.NewVecDense(outputs) // zero vector
	for i := 0; i < dLoss.Len(); i++ {
		v := dLossDB.AtVec(i)
		v += dLoss.AtVec(i) * s.activationFunction.D(mat.Dot(s.w.RowView(i), x) + s.b.AtVec(i))
		dLossDB.SetVec(i, v)
	}

	// calculate dLossDW
	dLossDW, _ := s.matArena.NewDense(outputs, inputs) // zero matrix
	dLossDW.Apply(func(i,j int, v float64)float64 {
		return v + dLoss.AtVec(i) *  s.activationFunction.D(mat.Dot(s.w.RowView(i), x) + s.b.AtVec(i)) * x.AtVec(j)
	}, dLossDW)

	// Calculate dLossDX
	dLossDX,_ := s.matArena.NewVecDense(x.Len())
	for j := 0; j < x.Len(); j++ {
		var sum float64
		for i := 0; i < dLoss.Len(); i++ {
			sum += dLoss.AtVec(i) *  s.activationFunction.D(mat.Dot(s.w.RowView(i), x) + s.b.AtVec(i)) * s.w.At(i,j)
		}
		dLossDX.SetVec(j, sum)
	}

	if alpha == 0 {
		return dLossDX
	}

	// learn
	dLossDW.Scale(-alpha, dLossDW)
	s.w.Add(s.w, dLossDW)
	dLossDW.Zero()

	dLossDB.ScaleVec(-alpha, dLossDB)
	s.b.AddVec(s.b, dLossDB)
	dLossDB.Zero()

	return dLossDX
}