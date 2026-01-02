package fn

import (
	"gonum.org/v1/gonum/mat"
)

// Serial returns a single layer from multiple layers executed one after another.
// TODO: I bet we can re-think this to be recursive. This would be quite elegant and simple.
func Serial(layers ...Layer) Layer {
	if len(layers) == 1 {
		return layers[0]
	}
	return ser2{
		left:  layers[0],
		right: Serial(layers[1:]...),
	}
	var n int
	for _, l := range layers {
		n += l.NumWeights()
	}

	return ser{
		layers:     layers,
		numWeights: n,
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

type ser struct {
	layers     []Layer
	numWeights int
}

func ident(n int) mat.Matrix {
	s := make([]float64, n)
	m := mat.NewDiagDense(n, s)
	for i := range s {
		s[i] = 1.
	}
	return m
}

func (s ser) D(x mat.Vector, h []float64) (mat.Matrix, mat.Matrix) {
	type node struct {
		dYdX mat.Matrix
		dYdW mat.Matrix
	}
	var nodes []node
	xPrime := x
	offset := 0
	for _, layer := range s.layers {
		n := layer.NumWeights()
		dYdX, dYdW := layer.D(xPrime, h[offset:(offset+n)])
		nodes = append(nodes, node{dYdX, dYdW})
		xPrime = layer.F(xPrime, h[offset:(offset+n)])
		offset += n
	}
	dYdX := ident(xPrime.Len())
	dYdW := mat.NewDense(xPrime.Len(), s.numWeights, nil)
	offset = 0
	for i := range nodes {
		node := nodes[len(nodes)-1-i]
		if node.dYdW != nil {
			var inter mat.Dense
			inter.Mul(dYdX, node.dYdW) // this is confusing and needs a comment
			_, c := node.dYdW.Dims()
			place(dYdW, 0, s.numWeights-offset-c, &inter)
			offset += c
		}
		var inter mat.Dense
		inter.Mul(dYdX, node.dYdX)
		dYdX = &inter
	}
	return dYdX, dYdW
}

func (s ser) F(x mat.Vector, h []float64) mat.Vector {
	xPrime := x
	offset := 0
	for _, l := range s.layers {
		n := l.NumWeights()
		xPrime = l.F(xPrime, h[offset:(offset+n)])
		offset += n
	}
	return xPrime
}

func (s ser) NumWeights() int {
	return s.numWeights
}
