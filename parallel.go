package fn

import (
	"gonum.org/v1/gonum/mat"
)

// Parallel returns a single layer from multiple layers executed independently. Each layer's input
// must be of the same cardinality. The output cardinality is the sum of the individual layer's outputs.
func Parallel(layers ...Layer) Layer {
	var n int
	for _, l := range layers {
		_, _, w := l.Shape()
		n += w
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
		_, _, numWeights := layer.Shape()
		j := offset + numWeights
		i := offset
		hs = append(hs, h[i:j])
		offset += numWeights
	}
	return hs
}

func (p par) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	hs := p.weights(h)
	// Pre-allocate temps with correct sizes
	temps := make([]*mat.VecDense, len(hs))
	for i, h := range hs {
		_, outputs, _ := p.layers[i].Shape()
		temps[i] = mat.NewVecDense(outputs, nil)
		p.layers[i].F(temps[i], x, h)
	}
	// Second pass: concatenate results
	var offset int
	for i := range temps {
		y := temps[i]
		for j := 0; j < y.Len(); j++ {
			dst.SetVec(offset+j, y.AtVec(j))
		}
		offset += y.Len()
	}
}

func (p par) D(dYdX *mat.Dense, dYdH *mat.Dense, x mat.Vector, h []float64) {
	hs := p.weights(h)

	// Allocate temporary storage for all derivatives
	dxTemps := make([]*mat.Dense, len(hs))
	dhTemps := make([]*mat.Dense, len(hs))

	var yLen int
	for i, h := range hs {
		layerInputs, layerOutputs, layerWeights := p.layers[i].Shape()
		dxTemps[i] = mat.NewDense(layerOutputs, layerInputs, nil)
		if dYdH != nil && layerWeights > 0 {
			dhTemps[i] = mat.NewDense(layerOutputs, layerWeights, nil)
			p.layers[i].D(dxTemps[i], dhTemps[i], x, h)
		} else {
			p.layers[i].D(dxTemps[i], nil, x, h)
		}
		yLen += layerOutputs
	}

	// Assemble dYdX
	dYdX.Zero()
	var offset int
	for i := range dxTemps {
		place(dYdX, offset, 0, dxTemps[i])
		r, _ := dxTemps[i].Dims()
		offset += r
	}

	// Assemble dYdH if requested
	if dYdH != nil {
		dYdH.Zero()
		var iOffset int
		var jOffset int
		for i := range dhTemps {
			if dhTemps[i] != nil {
				place(dYdH, iOffset, jOffset, dhTemps[i])
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

func (p par) Shape() (inputs, outputs, weights int) {
	if len(p.layers) == 0 {
		return 0, 0, 0
	}
	firstInputs, _, _ := p.layers[0].Shape()
	totalOutputs := 0
	totalWeights := 0
	for _, layer := range p.layers {
		_, layerOutputs, layerWeights := layer.Shape()
		totalOutputs += layerOutputs
		totalWeights += layerWeights
	}
	return firstInputs, totalOutputs, totalWeights
}
