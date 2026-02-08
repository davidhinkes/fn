// Package test has utilities for testing functional networks.
package test

import (
	"gonum.org/v1/gonum/mat"
)

type Truth interface {
	// Dims returns the function's dimensions: (input cardinality, output cardinality)
	Dims() (int, int)
	F(dst *mat.VecDense, x mat.Vector)
	Rand(dst *mat.VecDense)
}

func MakeExamples(t Truth, n int) (*mat.Dense, *mat.Dense) {
	inputDim, outputDim := t.Dims()
	X := mat.NewDense(n, inputDim, nil)
	Y := mat.NewDense(n, outputDim, nil)
	x := mat.NewVecDense(inputDim, nil)
	y := mat.NewVecDense(outputDim, nil)
	for i := 0; i < n; i++ {
		t.Rand(x)
		t.F(y, x)
		X.SetRow(i, x.RawVector().Data)
		Y.SetRow(i, y.RawVector().Data)
	}
	return X, Y
}
