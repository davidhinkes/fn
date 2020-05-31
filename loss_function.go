package fn

import (
	"gonum.org/v1/gonum/mat"
)

type LossFunction interface {
	// F returns the loss (first return argument) and partial
	// derivative with respect to y (second return argument).
	F(y mat.Vector, yHat mat.Vector) (float64, mat.Vector)
}
