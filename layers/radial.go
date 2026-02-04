package layers

import (
	"fn"

	"math"

	"gonum.org/v1/gonum/mat"
)

func Radial(outputs int) fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		return radial{
			inputs:  inputs,
			outputs: outputs,
		}
	}
}

type radial struct {
	inputs  int
	outputs int
}

func (r radial) Shape() (inputs, outputs, weights int) {
	return r.inputs, r.outputs, r.inputs * r.outputs
}

// For a given vector (from matrix h) w:
// compute ||x-w||
func (r radial) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	v := mat.NewDense(r.outputs, r.inputs, h)
	// d is a temp vector, re-use to save space
	var d mat.VecDense
	for i := 0; i < r.outputs; i++ {
		d.SubVec(x, v.RowView(i))
		dst.SetVec(i, math.Sqrt(mat.Dot(&d, &d)))
	}
}

// Note that this as been modified by Claude Code and I was not able to verify.
func (r radial) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64) {
	// y[i] = ||x - w_i||
	// dY[i]/dX[j] = (x[j] - w[i,j]) / y[i]
	// dY[i]/dW[i,j] = -(x[j] - w[i,j]) / y[i]
	y := mat.NewVecDense(r.outputs, nil)
	r.F(y, x, h)
	w := mat.NewDense(r.outputs, r.inputs, h)

	dLdX.Zero()
	for i := 0; i < r.outputs; i++ {
		f := dLdY.AtVec(i) / y.AtVec(i)
		for j := 0; j < r.inputs; j++ {
			val := x.AtVec(j) - w.At(i, j)
			// dLdX[j] += dLdY[i] * dY[i]/dX[j]
			dLdX.SetVec(j, dLdX.AtVec(j)+f*val)
			// dLdH[i*inputs + j] = dLdY[i] * dY[i]/dW[i,j]
			dLdH.SetVec(i*r.inputs+j, -f*val)
		}
	}
}

var _ fn.Layer = radial{}
