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
	n := s.left.NumWeights()
	dYdX, dYdAleph := s.left.D(x, h[:n])
	y := s.left.F(x, h[:n])
	dZdY, dZdBet := s.right.D(y, h[n:])
	var dZdX mat.Dense
	dZdX.Mul(dZdY, dYdX)
	// it's possible that dZdAleph and/or dZdBet are nil
	if dYdAleph == nil && dZdBet == nil {
		return &dZdX, nil
	}
	if dYdAleph == nil {
		return &dZdX, dZdBet
	}
	var dZdAleph mat.Dense
	dZdAleph.Mul(dZdY, dYdAleph)
	if dZdBet == nil {
		return &dZdX, &dZdAleph
	}
	var dZdH mat.Dense
	dZdH.Augment(&dZdAleph, dZdBet)

	return &dZdX, &dZdH
}