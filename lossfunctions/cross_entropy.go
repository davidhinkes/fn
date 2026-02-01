package lossfunctions

import (
	"math"

	"fn"

	"gonum.org/v1/gonum/mat"
)

func NewCrossEntropy() fn.LossFunction {
	return crossEntropy{}
}

type crossEntropy struct {
}

func (c crossEntropy) F(dst *mat.VecDense, y mat.Vector, yHat mat.Vector) float64 {
	n := y.Len()
	var loss float64
	for i := 0; i < n; i++ {
		yi := math.Max(y.AtVec(i), 1e-15)
		yHati := yHat.AtVec(i)
		loss -= yHati * math.Log(yi)
		dst.SetVec(i, -yHati/yi)
	}
	return loss
}
