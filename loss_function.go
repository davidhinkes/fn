package fn

import (
	"gonum.org/v1/gonum/mat"
)

type LossFunction interface {
	F(y mat.Vector, yHat mat.Vector) (float64, mat.Vector)
}
