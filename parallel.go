package fn

import (
	"gonum.org/v1/gonum/mat"
)

// Parallel returns a single layer from multiple layers executed independently. Each layer's input
// must be of the same cardinality. The output cardinality is the sum of the individual layer's outputs.
func Parallel(layers ...Layer) Layer {
	return par{
		layers: layers,
	}
}

func place(dst *mat.Dense, i int, j int, m mat.Matrix) {
	r, c := m.Dims()
	for k := 0; k < r; k++ {
		for l := 0; l < c; l++ {
			dst.Set(k+i, l+j, m.At(k, l))
		}
	}
}

type par struct {
	layers []Layer
}

func (p par) hyperparameters(h []float64) [][]float64 {
	var hs [][]float64
	var offset int
	for _, layer := range p.layers {
		numHyperparameters := layer.NumHyperparameters()
		j := offset + numHyperparameters
		i := offset
		hs = append(hs, h[i:j])
		offset += numHyperparameters
	}
	return hs
}

func (p par) F(x mat.Vector, h []float64) mat.Vector {
	hs := p.hyperparameters(h)
	var yLen int
	var ys []mat.Vector
	for i, h := range hs {
		y := p.layers[i].F(x, h)
		yLen += y.Len()
		ys = append(ys, y)
	}
	ret := make([]float64, yLen)
	var offset int
	for _, y := range ys {
		i := offset
		j := offset + y.Len()
		mat.Col(ret[i:j], 0, y)
		offset += y.Len()
	}
	return mat.NewVecDense(yLen, ret)
}

func (p par) D(x mat.Vector, h []float64) (mat.Matrix, mat.Matrix) {
	hs := p.hyperparameters(h)
	var dxs []mat.Matrix
	var dhs []mat.Matrix
	var yLen int
	for i, h := range hs {
		dx, dh := p.layers[i].D(x, h)
		dxs = append(dxs, dx)
		dhs = append(dhs, dh)
		{
			y, _ := dh.Dims()
			yLen += y
		}
	}
	dydx := mat.NewDense(yLen, x.Len(), nil)
	var offset int
	for _, m := range dxs {
		place(dydx, offset, 0, m)
		r, _ := m.Dims()
		offset += r
	}
	dydh := mat.NewDense(yLen, len(h), nil)
	var iOffset int
	var jOffset int
	for _, m := range dhs {
		place(dydh, iOffset, jOffset, m)
		{
			r, c := m.Dims()
			iOffset += r
			jOffset += c
		}
	}

	return dydx, dydh
}

func (p par) NumHyperparameters() int {
	var sum int
	for _, layer := range p.layers {
		sum += layer.NumHyperparameters()
	}
	return sum
}
