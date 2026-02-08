package layers

import (
	"fn"
	"fn/matrixpool"

	"gonum.org/v1/gonum/mat"
)

func Serial(builders ...fn.LayerBuilder) fn.LayerBuilder {
	return func(inputs int) fn.Layer {
		leftLayer := builders[0](inputs)
		if len(builders) == 1 {
			return leftLayer
		}
		_, leftLayerOutputs, _ := leftLayer.Shape()
		return serialNode{
			left:  leftLayer,
			right: Serial(builders[1:]...)(leftLayerOutputs),
		}
	}
}

// Type serialNode is the core struct that implements a serial Layer.
// It is implemented via recursion, which is elegant but perhaps has performance
// concerns.
type serialNode struct {
	left  fn.Layer
	right fn.Layer
}

func (s serialNode) Shape() (inputs, outputs, weights int) {
	leftInputs, _, leftWeights := s.left.Shape()
	_, rightOutputs, rightWeights := s.right.Shape()
	return leftInputs, rightOutputs, leftWeights + rightWeights
}

func (s serialNode) F(dst *mat.Dense, X mat.Matrix, h []float64) {
	rows, _ := X.Dims()
	_, leftOutputs, leftWeights := s.left.Shape()
	xPrime := matrixpool.GetDense(rows, leftOutputs)
	defer matrixpool.PutDense(xPrime)
	s.left.F(xPrime, X, h[:leftWeights])
	s.right.F(dst, xPrime, h[leftWeights:])
}

func (s serialNode) D(dLdX *mat.Dense, dLdH *mat.VecDense, dLdZ mat.Matrix, X mat.Matrix, h []float64) {
	// z = right(left(x))
	// [ℵ, ℶ] = h
	// want: dLdℵ and dLdℶ (these will make up dLdH)
	// want: dLdX
	// Backward pass: propagate dLdZ through right to get dLdY, then through left to get dLdX
	rows, _ := X.Dims()
	_, leftOutputs, leftWeights := s.left.Shape()
	_, _, rightWeights := s.right.Shape()

	// Forward pass to get intermediate value y = left(x)
	y := matrixpool.GetDense(rows, leftOutputs)
	defer matrixpool.PutDense(y)
	s.left.F(y, X, h[:leftWeights])

	// When dLdH is non-nil, back sub-layer weight gradient vectors directly
	// by slices of dLdH's data — sub-layers write straight into dLdH.
	var dLdℵ, dLdℶ *mat.VecDense
	if dLdH != nil {
		raw := dLdH.RawVector().Data
		if leftWeights > 0 {
			dLdℵ = mat.NewVecDense(leftWeights, raw[:leftWeights])
		}
		if rightWeights > 0 {
			dLdℶ = mat.NewVecDense(rightWeights, raw[leftWeights:])
		}
	}

	// Backward through right layer: dLdZ -> dLdY
	dLdY := matrixpool.GetDense(rows, leftOutputs)
	defer matrixpool.PutDense(dLdY)
	s.right.D(dLdY, dLdℶ, dLdZ, y, h[leftWeights:])

	// Backward through left layer: dLdY -> dLdX
	s.left.D(dLdX, dLdℵ, dLdY, X, h[:leftWeights])
}
