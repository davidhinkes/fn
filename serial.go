package fn

import (
	"gonum.org/v1/gonum/mat"
)

// Serial returns a single layer from multiple layers executed one after another.
func Serial(layers ...Layer) Layer {
	if len(layers) == 1 {
		return layers[0]
	}
	return ser2{
		left:  layers[0],
		right: Serial(layers[1:]...),
	}
}

// Type ser2 is the serial type, that adheres to the Layer interface.
// It is implemented via recursion, which is elegant but perhaps has performance
// concerns.
type ser2 struct {
	left  Layer
	right Layer
}

func (s ser2) NumWeights() int {
	return s.left.NumWeights() + s.right.NumWeights()
}

func (s ser2) F(x mat.Vector, h []float64) mat.Vector {
	n := s.left.NumWeights()
	xPrime := s.left.F(x, h[:n])
	return s.right.F(xPrime, h[n:])
}

func (s ser2) D(x mat.Vector, h []float64) (mat.Matrix, mat.Matrix) {
	// [ℵ, ℶ] = h
	// y = left(x, ℵ)
	// z = right(y, ℶ)
	// known: dZdY, dYdX, dYdℵ, and dZdℶ (via left/right F & D)
	// want: dZdX, and dZdH
	// dZdH = [dZdℵ, dZdℶ]
	// dZdℵ = dZdY * dYdℵ (matrix multiplication)
	// dZdX = dZdY * dYdX
	n := s.left.NumWeights()
	dYdX, dYdℵ := s.left.D(x, h[:n])
	y := s.left.F(x, h[:n])
	dZdY, dZdℶ := s.right.D(y, h[n:])
	var dZdX mat.Dense
	dZdX.Mul(dZdY, dYdX)
	// it's possible that dZdℵ and/or dZdℶ are nil
	if dYdℵ == nil && dZdℶ == nil {
		return &dZdX, nil
	}
	if dYdℵ == nil {
		return &dZdX, dZdℶ
	}
	var dZdℵ mat.Dense
	dZdℵ.Mul(dZdY, dYdℵ)
	if dZdℶ == nil {
		return &dZdX, &dZdℵ
	}
	var dZdH mat.Dense
	dZdH.Augment(&dZdℵ, dZdℶ)

	return &dZdX, &dZdH
}
