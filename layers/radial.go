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

func (r radial) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	v := mat.NewDense(r.outputs, r.inputs, h)
	dst.ReuseAsVec(r.outputs)
	// d is a temp vector, re-use to save space
	var d mat.VecDense
	for i := 0; i < r.outputs; i++ {
		d.SubVec(x, v.RowView(i))
		dst.SetVec(i, math.Sqrt(mat.Dot(&d, &d)))
	}
}

func (r radial) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64) {
	var y mat.VecDense
	r.F(&y, x, h)
	w := mat.NewDense(r.outputs, r.inputs, h)

	dYdX.Reset()
	dYdX.ReuseAs(r.outputs, r.inputs)

	if dYdH != nil {
		dYdH.Reset()
		dYdH.ReuseAs(r.outputs, r.NumWeights())
	}

	for i := 0; i < r.outputs; i++ {
		f := 1. / y.AtVec(i)
		for j := 0; j < r.inputs; j++ {
			k := i*r.inputs + j // k is the index of Wij into h
			val := x.AtVec(j) - w.At(i, j)
			if dYdH != nil {
				dYdH.Set(i, k, -f*val)
			}
			dYdX.Set(i, j, f*val)
		}
	}
}

var _ fn.Layer = radial{}
