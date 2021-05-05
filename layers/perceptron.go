package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

func MakePerceptronLayer(inputs, outputs int) fn.Layer {
	return &perceptron{
		inputs:  inputs,
		outputs: outputs,
	}
}

type perceptron struct {
	inputs  int
	outputs int
}

func (p *perceptron) dim(x int) (int, int) {
	c := p.inputs
	return x / c, x % c
}

func (p *perceptron) mkWeights(h []float64) mat.Matrix {
	return mat.NewDense(p.outputs, p.inputs, h)
}

func (p *perceptron) D(x mat.Vector, h []float64) (mat.Matrix, mat.Matrix) {
	w := p.mkWeights(h)
	dYdH := mat.NewDense(p.outputs, len(h), nil)
	dYdH.Apply(func(i, j int, _ float64) float64 {
		l, m := p.dim(j)
		if i != l {
			return 0
		}
		return x.AtVec(m)
	}, dYdH)
	return w, dYdH
}

func (p *perceptron) F(x mat.Vector, h []float64) mat.Vector {
	var ret mat.VecDense
	ret.MulVec(p.mkWeights(h), x)
	return &ret
}

func (p *perceptron) NumWeights() int {
	return p.inputs * p.outputs
}
