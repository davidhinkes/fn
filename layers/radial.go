package layers

import (
	"fn"

	"math"

	"gonum.org/v1/gonum/mat"
)

func Radial(outputs int) fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return wrapVectorLayer(radial{
			inputs:  inputs,
			outputs: outputs,
		})
	}
}

type radial struct {
	inputs  int
	outputs int
}

func (r radial) Shape() (inputs, outputs, weights int) {
	return r.inputs, r.outputs, r.inputs * r.outputs
}

func (r radial) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	w := mat.NewDense(r.outputs, r.inputs, h)
	for i := 0; i < r.outputs; i++ {
		var sum float64
		for j := 0; j < r.inputs; j++ {
			d := x.AtVec(j) - w.At(i, j)
			sum += d * d
		}
		dst.SetVec(i, math.Sqrt(sum))
	}
}

func (r radial) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64) {
	// Recompute forward output.
	y := mat.NewVecDense(r.outputs, nil)
	r.F(y, x, h)

	w := mat.NewDense(r.outputs, r.inputs, h)

	// Zero dLdX — we accumulate across output centers.
	for j := 0; j < r.inputs; j++ {
		dLdX.SetVec(j, 0)
	}
	for i := 0; i < r.outputs; i++ {
		f := dLdY.AtVec(i) / y.AtVec(i)
		for j := 0; j < r.inputs; j++ {
			val := x.AtVec(j) - w.At(i, j)
			dLdX.SetVec(j, dLdX.AtVec(j)+f*val)
			idx := i*r.inputs + j
			dLdH.SetVec(idx, -f*val)
		}
	}
}

var _ VectorLayer = radial{}
