package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

func Perceptron(outputs int) fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return &perceptron{
			inputs:  inputs,
			outputs: outputs,
		}
	}
}

type perceptron struct {
	inputs  int
	outputs int
}

func (p *perceptron) mkWeights(h []float64) mat.Matrix {
	return mat.NewDense(p.outputs, p.inputs, h)
}

func (p *perceptron) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64) {
	w := p.mkWeights(h)
	dYdX.Copy(w)
	rows, columns := w.Dims()
	dYdH.Zero()
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			// assumption of row-major layout of h & w
			dYdH.Set(i, columns*i+j, x.AtVec(j))
		}
	}
}

func (p *perceptron) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	dst.MulVec(p.mkWeights(h), x)
}

func (p *perceptron) Shape() (inputs, outputs, weights int) {
	return p.inputs, p.outputs, p.inputs * p.outputs
}
