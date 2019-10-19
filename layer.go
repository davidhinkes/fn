package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Model []Layer

type Layer interface {
	F(x mat.Vector) mat.Vector
	D(x mat.Vector, dLoss mat.Vector) mat.Vector
	Learn(alpha float64)
}
