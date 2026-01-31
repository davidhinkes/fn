package fn

import (
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

func (s serialNode) NumWeights() int {
	return s.left.NumWeights() + s.right.NumWeights()
}

func (s serialNode) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	n := s.left.NumWeights()
	var xPrime mat.VecDense
	s.left.F(&xPrime, x, h[:n])
	s.right.F(dst, &xPrime, h[n:])
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
	n := s.left.NumWeights()

	// Compute y = left(x)
	var y mat.VecDense
	s.left.F(&y, x, h[:n])

	// Get derivatives from left layer
	var dYdX mat.Dense
	var dYdℵ mat.Dense

	s.left.D(&dYdX, &dYdℵ, x, h[:n])

	// Get derivatives from right layer
	var dZdY mat.Dense
	var dZdℶ mat.Dense
	s.right.D(&dZdY, &dZdℶ, &y, h[n:])

	// Compute dZdX = dZdY * dYdX
	dZdX.Mul(&dZdY, &dYdX)
	hasLeftWeights, hasRightWeights := s.left.NumWeights() > 0, s.right.NumWeights() > 0
	if !hasLeftWeights && !hasRightWeights {
		// We're done, don't bother computing dZdH
		return
	}
	if !hasLeftWeights {
		// This is simple, just dZdH == dZdℶ
		dZdH.CloneFrom(&dZdℶ)
		return
	}
	// At this point, we need dZdℵ = dZdY * dYdℵ
	var dZdℵ mat.Dense
	dZdℵ.Mul(&dZdY, &dYdℵ)
	if !hasRightWeights {
		// Only left layer has weights
		dZdH.CloneFrom(&dZdℵ)
		return
	}
	// Both layers have weights, concatenate horizontally
	dZdH.Augment(&dZdℵ, &dZdℶ)
}
