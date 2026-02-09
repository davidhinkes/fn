package layers

import (
	"fn"
	"fn/matrixpool"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// VectorLayer is a simpler per-vector layer interface for layers that
// don't benefit from batch matrix operations. The vectorLayerConverter
// adapter wraps a VectorLayer to satisfy the batch fn.Layer interface.
type VectorLayer interface {
	F(dst *mat.VecDense, x mat.Vector, h []float64)
	D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdY mat.Vector, x mat.Vector, h []float64)
	Shape() (inputs, outputs, weights int)
}

type vectorLayerConverter struct {
	vl VectorLayer
}

func wrapVectorLayer(vl VectorLayer) fn.Layer {
	return vectorLayerConverter{vl: vl}
}

func (c vectorLayerConverter) Shape() (inputs, outputs, weights int) {
	return c.vl.Shape()
}

func (c vectorLayerConverter) F(dst *mat.Dense, X mat.Matrix, h []float64) {
	rows, cols := X.Dims()
	xVec := matrixpool.GetVec(cols)
	defer matrixpool.PutVec(xVec)
	xBuf := xVec.RawVector().Data
	var dstRow mat.VecDense
	for row := 0; row < rows; row++ {
		dstRow.RowViewOf(dst, row)
		mat.Row(xBuf, row, X)
		c.vl.F(&dstRow, xVec, h)
	}
}

func (c vectorLayerConverter) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdY mat.Matrix, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	inputs, outputs, weights := c.vl.Shape()

	// Zero dLdH before accumulating across rows.
	if dLdH != nil && weights > 0 {
		dLdH.Zero()
	}

	// Pool scratch vectors for extracting rows.
	xVec := matrixpool.GetVec(inputs)
	defer matrixpool.PutVec(xVec)

	dLdYVec := matrixpool.GetVec(outputs)
	defer matrixpool.PutVec(dLdYVec)

	var tmpDH *mat.VecDense
	if dLdH != nil && weights > 0 {
		tmpDH = matrixpool.GetVec(weights)
		defer matrixpool.PutVec(tmpDH)
	}

	var dLdXRow mat.VecDense
	for row := 0; row < rows; row++ {
		dLdXRow.RowViewOf(dLdX, row)
		mat.Row(xVec.RawVector().Data, row, X)
		mat.Row(dLdYVec.RawVector().Data, row, dLdY)

		if tmpDH != nil {
			c.vl.D(&dLdXRow, tmpDH, dLdYVec, xVec, h)
			floats.Add(dLdH.RawVector().Data, tmpDH.RawVector().Data)
		} else {
			c.vl.D(&dLdXRow, nil, dLdYVec, xVec, h)
		}
	}
}
