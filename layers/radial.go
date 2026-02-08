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

func (r radial) F(dst *mat.Dense, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	w := mat.NewDense(r.outputs, r.inputs, h)
	for row := 0; row < rows; row++ {
		for i := 0; i < r.outputs; i++ {
			var sum float64
			for j := 0; j < r.inputs; j++ {
				d := X.At(row, j) - w.At(i, j)
				sum += d * d
			}
			dst.Set(row, i, math.Sqrt(sum))
		}
	}
}

func (r radial) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	// Recompute forward output.
	Y := mat.NewDense(rows, r.outputs, nil)
	r.F(Y, X, h)

	w := mat.NewDense(r.outputs, r.inputs, h)

	// Zero dLdH — we accumulate across batch rows.
	for i := 0; i < dLdH.Len(); i++ {
		dLdH.SetVec(i, 0)
	}

	for row := 0; row < rows; row++ {
		// Zero this row of dLdX — we accumulate across output centers.
		for j := 0; j < r.inputs; j++ {
			dLdX.Set(row, j, 0)
		}
		for i := 0; i < r.outputs; i++ {
			f := dLdY.At(row, i) / Y.At(row, i)
			for j := 0; j < r.inputs; j++ {
				val := X.At(row, j) - w.At(i, j)
				dLdX.Set(row, j, dLdX.At(row, j)+f*val)
				idx := i*r.inputs + j
				dLdH.SetVec(idx, dLdH.AtVec(idx)-f*val)
			}
		}
	}
}

var _ fn.Layer = radial{}
