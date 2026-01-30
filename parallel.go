package fn

import (
	"gonum.org/v1/gonum/mat"
)

// Parallel returns a single layer from multiple layers executed independently. Each layer's input
// must be of the same cardinality. The output cardinality is the sum of the individual layer's outputs.
func Parallel(layers ...Layer) Layer {
	var n int
	for _, l := range layers {
		n += l.NumWeights()
	}

	return par{
		layers:     layers,
		numWeights: n,
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
	layers     []Layer
	numWeights int
}

func (p par) weights(h []float64) [][]float64 {
	var hs [][]float64
	var offset int
	for _, layer := range p.layers {
		numWeights := layer.NumWeights()
		j := offset + numWeights
		i := offset
		hs = append(hs, h[i:j])
		offset += numWeights
	}
	return hs
}

func (p par) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	hs := p.weights(h)
	var yLen int
	// First pass: compute all outputs and accumulate total length
	temps := make([]mat.VecDense, len(hs))
	for i, h := range hs {
		p.layers[i].F(&temps[i], x, h)
		yLen += temps[i].Len()
	}
	// Second pass: concatenate results
	dst.ReuseAsVec(yLen)
	var offset int
	for i := range temps {
		y := &temps[i]
		for j := 0; j < y.Len(); j++ {
			dst.SetVec(offset+j, y.AtVec(j))
		}
		offset += y.Len()
	}
}

func (p par) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64) {
	hs := p.weights(h)

	// Allocate temporary storage for all derivatives
	dxTemps := make([]mat.Dense, len(hs))
	dhTemps := make([]mat.Dense, len(hs))

	var yLen int
	for i, h := range hs {
		if dYdH != nil && p.layers[i].NumWeights() > 0 {
			p.layers[i].D(&dxTemps[i], &dhTemps[i], x, h)
		} else {
			p.layers[i].D(&dxTemps[i], nil, x, h)
		}
		r, _ := dxTemps[i].Dims()
		yLen += r
	}

	// Assemble dYdX
	dYdX.Reset()
	dYdX.ReuseAs(yLen, x.Len())
	var offset int
	for i := range dxTemps {
		place(dYdX, offset, 0, &dxTemps[i])
		r, _ := dxTemps[i].Dims()
		offset += r
	}

	// Assemble dYdH if requested
	if dYdH != nil {
		dYdH.Reset()
		dYdH.ReuseAs(yLen, len(h))
		var iOffset int
		var jOffset int
		for i := range dhTemps {
			if p.layers[i].NumWeights() > 0 {
				place(dYdH, iOffset, jOffset, &dhTemps[i])
				r, c := dhTemps[i].Dims()
				iOffset += r
				jOffset += c
			} else {
				r, _ := dxTemps[i].Dims()
				iOffset += r
			}
		}
	}
}

func (p par) NumWeights() int {
	return p.numWeights
}
