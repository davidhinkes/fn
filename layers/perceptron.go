package layers

import (
	"fn"

	"gonum.org/v1/gonum/mat"
)

func makeHyperparameters(inputs, outputs int) (*mat.Dense, *mat.VecDense) {
	n := inputs * outputs
	s := make([]float64, n)
	return mat.NewDense(outputs, inputs, s), mat.NewVecDense(n, s)
}

func MakePerceptronLayer(inputs, outputs int) fn.Layer {
	w, h := makeHyperparameters(inputs, outputs)
	randomize(h)
	return &perceptron{
		hyperparameters: h,
		w:               w,
	}
}

func (p *perceptron) dim(x int) (int, int) {
	_, c := p.w.Dims()
	return x / c, x % c
}

type perceptron struct {
	hyperparameters *mat.VecDense
	w               *mat.Dense
}

func (p *perceptron) D(x mat.Vector) (mat.Matrix, mat.Matrix) {
	m, _ := p.w.Dims()
	dYdH := mat.NewDense(m, p.hyperparameters.Len(), nil)
	dYdH.Apply(func(i, j int, _ float64) float64 {
		l, m := p.dim(j)
		if i != l {
			return 0
		}
		return x.AtVec(m)
	}, dYdH)
	return p.w, dYdH
}

func (p *perceptron) F(x mat.Vector) mat.Vector {
	var ret mat.VecDense
	ret.MulVec(p.w, x)
	return &ret
}

func (p *perceptron) Hyperparameters() *mat.VecDense {
	return p.hyperparameters
}
