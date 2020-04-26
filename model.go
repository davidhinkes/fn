package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Layers []Layer
	LossFunction LossFunction
}

// A layer represents a mathematical function.
// y = F(x)
type Layer interface {
	// F is the layer's forward function. Given vector x as an input, returns an output vector.
	// |return| = |y|
	F(x mat.Vector) mat.Vector
	// Learn updates the internal layer's hyperparameters and backpropogates partial derivatives.
	// Given the layer input, x, and the dLossDY, compute dLoss/dX
	// |dLossDY| = |y|
	// |return| = |x|
	// if alpha is 0, no learning is performed
	Learn(x mat.Vector, dLossDY mat.Vector, alpha float64) mat.Vector
}

func (m Model) Eval(x mat.Vector) (mat.Vector, []mat.Vector) {
	var upsilons []mat.Vector
	previousUpsilon := x
	for _, l := range m.Layers {
		u := l.F(previousUpsilon)
		upsilons = append(upsilons, u)
		previousUpsilon = u
	}
	return previousUpsilon, upsilons
}
