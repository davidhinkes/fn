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

func (p *perceptron) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64) {
	w := p.mkWeights(h)
	// dLdX = W^T * dLdY
	dLdX.MulVec(w.T(), dLdY)
	// dLdH: gradient w.r.t. W_ij is dLdY[i] * x[j]
	// h is row-major, so W_ij is at index h[i*inputs + j]
	for i := 0; i < p.outputs; i++ {
		dLdYi := dLdY.AtVec(i)
		for j := 0; j < p.inputs; j++ {
			// Confirmed this is correct via chain rule.
			// Note that all of h is covered. No need to Zero()
			dLdH.SetVec(i*p.inputs+j, dLdYi*x.AtVec(j))
		}
	}
}

func (p *perceptron) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	dst.MulVec(p.mkWeights(h), x)
}

func (p *perceptron) Shape() (inputs, outputs, weights int) {
	return p.inputs, p.outputs, p.inputs * p.outputs
}
