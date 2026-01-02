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

func (p *perceptron) mkWeights(h []float64) mat.Matrix {
	return mat.NewDense(p.outputs, p.inputs, h)
}

func (p *perceptron) D(x mat.Vector, h []float64) (mat.Matrix, mat.Matrix) {
	w := p.mkWeights(h)
	dYdH := mat.NewDense(p.outputs, len(h), nil)
	dYdH.Zero() // This is perhaps not needed, but I feel better.
	rows, columns := w.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			// assumption of row-major layout of h & w
			dYdH.Set(i, columns*i+j, x.AtVec(j))
		}
	}
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
