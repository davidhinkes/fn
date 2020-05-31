package fn

import (
	"gonum.org/v1/gonum/mat"
)

// A layer represents a mathematical function.
// y = F(x)
type Layer interface {
	// F is the layer's forward function. Given vector x as an input, returns an output vector.
	// |return| = |y|
	F(x mat.Vector) mat.Vector

	// Backpropagate partial derivatives
	// Given the layer input, x, and the dLossDY, compute dLoss/dx and
	// dLoss/dHyperparameters.
	// This function may be called from multiple go routines and thus should
	// be stateless.
	Backpropagate(x mat.Vector, dLossDY mat.Vector) (dLossDX mat.Vector, dLossDH mat.Vector)

	// Learn is expected to add v to the hyperparameters.
	Learn(v mat.Vector)
}
