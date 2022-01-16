package layers

import (
	"fn"

	"math"

	"gonum.org/v1/gonum/mat"
)

func MakeRadialLayer(inputs, outputs int) fn.Layer {
	return radial{
		inputs:  inputs,
		outputs: outputs,
	}
}

type radial struct {
	inputs  int
	outputs int
}

func (r radial) NumWeights() int {
	return r.inputs * r.outputs
}

func (r radial) F(x mat.Vector, h []float64) mat.Vector {
	v := mat.NewDense(r.outputs, r.inputs, h)
	ret := mat.NewVecDense(r.outputs, nil)
	// d is a temp vector, re-use to save space
	var d mat.VecDense
	for i := 0; i < r.outputs; i++ {
		d.SubVec(x, v.RowView(i))
		ret.SetVec(i, math.Sqrt(mat.Dot(&d, &d)))
	}
	return ret
}

func (r radial) D(x mat.Vector, h []float64) (mat.Matrix, mat.Matrix) {
	y := r.F(x, h)
	w := mat.NewDense(r.outputs, r.inputs, h)
	dYdX := mat.NewDense(r.outputs, r.inputs, nil)
	dYdW := mat.NewDense(r.outputs, r.NumWeights(), nil)
	for i := 0; i < r.outputs; i++ {
		f := 1. / y.AtVec(i)
		for j := 0; j < r.inputs; j++ {
			k := i*r.inputs + j // k is the index of Wij into h
			dYdW.Set(i, k, -f*(x.AtVec(j)-w.At(i, j)))
			dYdX.Set(i, j, f*(x.AtVec(j)-w.At(i, j)))
		}
	}
	return dYdX, dYdW
}

var _ fn.Layer = radial{}
