package fn

import (
	"fn/matrixpool"

	"gonum.org/v1/gonum/mat"
)

// Serial returns a single layer from multiple layers executed one after another.
func Serial(layers ...Layer) Layer {
	if len(layers) == 1 {
		return layers[0]
	}
	return serialNode{
		left:  layers[0],
		right: Serial(layers[1:]...),
	}
}

// Type serialNode is the core struct that implements a serial Layer.
// It is implemented via recursion, which is elegant but perhaps has performance
// concerns.
type serialNode struct {
	left  Layer
	right Layer
}

func (s serialNode) Shape() (inputs, outputs, weights int) {
	leftInputs, _, leftWeights := s.left.Shape()
	_, rightOutputs, rightWeights := s.right.Shape()
	return leftInputs, rightOutputs, leftWeights + rightWeights
}

func (s serialNode) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	_, leftOutputs, leftWeights := s.left.Shape()
	xPrime := matrixpool.GetVec(leftOutputs)
	defer matrixpool.PutVec(xPrime)
	s.left.F(xPrime, x, h[:leftWeights])
	s.right.F(dst, xPrime, h[leftWeights:])
}

func (s serialNode) D(dZdX *mat.Dense, dZdH *mat.Dense, x mat.Vector, h []float64) {
	// [ℵ, ℶ] = h
	// y = left(x, ℵ)
	// z = right(y, ℶ)
	// known: dZdY, dYdX, dYdℵ, and dZdℶ (via left/right F & D)
	// want: dZdX, and dZdH
	// dZdH = [dZdℵ, dZdℶ]
	// dZdℵ = dZdY * dYdℵ (matrix multiplication)
	// dZdX = dZdY * dYdX (matrix multiplication)
	leftInputs, leftOutputs, leftWeights := s.left.Shape()

	// Compute y = left(x)
	y := matrixpool.GetVec(leftOutputs)
	defer matrixpool.PutVec(y)
	s.left.F(y, x, h[:leftWeights])

	// Get derivatives from left layer
	dYdX := matrixpool.GetDense(leftOutputs, leftInputs)
	defer matrixpool.PutDense(dYdX)
	var dYdℵ *mat.Dense
	if leftWeights > 0 {
		dYdℵ = matrixpool.GetDense(leftOutputs, leftWeights)
		defer matrixpool.PutDense(dYdℵ)
	}

	s.left.D(dYdX, dYdℵ, x, h[:leftWeights])

	// Get derivatives from right layer
	rightInputs, rightOutputs, rightWeights := s.right.Shape()
	dZdY := matrixpool.GetDense(rightOutputs, rightInputs)
	defer matrixpool.PutDense(dZdY)
	var dZdℶ *mat.Dense
	if rightWeights > 0 {
		dZdℶ = matrixpool.GetDense(rightOutputs, rightWeights)
		defer matrixpool.PutDense(dZdℶ)
	}
	s.right.D(dZdY, dZdℶ, y, h[leftWeights:])

	// Compute dZdX = dZdY * dYdX
	dZdX.Mul(dZdY, dYdX)
	if leftWeights == 0 && rightWeights == 0 {
		// We're done, don't bother computing dZdH
		return
	}
	if leftWeights == 0 {
		// This is simple, just dZdH == dZdℶ
		dZdH.CloneFrom(dZdℶ)
		return
	}
	// At this point, we need dZdℵ = dZdY * dYdℵ
	dZdℵ := matrixpool.GetDense(rightOutputs, leftWeights)
	defer matrixpool.PutDense(dZdℵ)
	dZdℵ.Mul(dZdY, dYdℵ)
	if rightWeights == 0 {
		// Only left layer has weights
		dZdH.CloneFrom(dZdℵ)
		return
	}
	// Both layers have weights, concatenate horizontally
	dZdH.Augment(dZdℵ, dZdℶ)
}
