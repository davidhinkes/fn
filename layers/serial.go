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

func (s serialNode) F(dst *mat.VecDense, x mat.Vector, h []float64) {
	_, leftOutputs, leftWeights := s.left.Shape()
	xPrime := matrixpool.GetVec(leftOutputs)
	defer matrixpool.PutVec(xPrime)
	s.left.F(xPrime, x, h[:leftWeights])
	s.right.F(dst, xPrime, h[leftWeights:])
}

func (s serialNode) D(dLdX *mat.VecDense, dLdH *mat.VecDense, dLdZ mat.Vector, x mat.Vector, h []float64) {
	// z = right(left(x))
	// [ℵ, ℶ] = h
	// want: dLdℵ and dLdℶ (these will make up dLdH)
	// want: dLdX
	// Backward pass: propagate dLdZ through right to get dLdY, then through left to get dLdX
	_, leftOutputs, leftWeights := s.left.Shape()
	_, _, rightWeights := s.right.Shape()

	// Forward pass to get intermediate value y = left(x)
	y := matrixpool.GetVec(leftOutputs)
	defer matrixpool.PutVec(y)
	s.left.F(y, x, h[:leftWeights])

	// Backward through right layer: dLdZ -> dLdY
	dLdY := matrixpool.GetVec(leftOutputs)
	defer matrixpool.PutVec(dLdY)
	var dLdℶ *mat.VecDense
	if rightWeights > 0 {
		dLdℶ = matrixpool.GetVec(rightWeights)
		defer matrixpool.PutVec(dLdℶ)
	}
	s.right.D(dLdY, dLdℶ, dLdZ, y, h[leftWeights:])

	// Backward through left layer: dLdY -> dLdX
	var dLdℵ *mat.VecDense
	if leftWeights > 0 {
		dLdℵ = matrixpool.GetVec(leftWeights)
		defer matrixpool.PutVec(dLdℵ)
	}
	s.left.D(dLdX, dLdℵ, dLdY, x, h[:leftWeights])

	// Concatenate weight gradients: dLdH = [dLdℵ, dLdℶ]
	// Ideally, the two sub-matrixes could just be pointers into dLdH
	if dLdH == nil {
		return
	}
	if leftWeights > 0 {
		for i := 0; i < leftWeights; i++ {
			dLdH.SetVec(i, dLdℵ.AtVec(i))
		}
	}
	if rightWeights > 0 {
		for i := 0; i < rightWeights; i++ {
			dLdH.SetVec(leftWeights+i, dLdℶ.AtVec(i))
		}
	}
}
