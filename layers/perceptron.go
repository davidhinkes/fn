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

func (p *perceptron) mkWeights(h []float64) *mat.Dense {
	return mat.NewDense(p.outputs, p.inputs, h)
}

func (p *perceptron) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64) {
	W := p.mkWeights(h)
	// dLdX = dLdY · W          (B×n) = (B×m)·(m×n)
	dLdX.Mul(dLdY, W)
	// dLdH = dLdYᵀ · X         (m×n) = (m×B)·(B×n)
	// This single GEMM replaces B outer products + accumulation.
	// Interpret dLdH's flat backing slice as an (m×n) Dense so the
	// multiply writes directly into dLdH with no copy.
	dLdHMat := mat.NewDense(p.outputs, p.inputs, dLdH.RawVector().Data)
	dLdHMat.Mul(dLdY.T(), X)
}

func (p *perceptron) F(dst *mat.Dense, X mat.Matrix, h []float64) {
	// Y = X · Wᵀ               (B×m) = (B×n)·(n×m)
	dst.Mul(X, p.mkWeights(h).T())
}

func (p *perceptron) Shape() (inputs, outputs, weights int) {
	return p.inputs, p.outputs, p.inputs * p.outputs
}
