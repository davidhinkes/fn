// Package test has utlities for testing functional networks.
package test

import (
	"gonum.org/v1/gonum/mat"
)

type Truth interface {
	// Dims returns the function's dimentions: (input cardinality, output cardinality)
	Dims() (int, int)
	F(dst *mat.VecDense, x mat.Vector)
	Rand(dst *mat.VecDense)
}

func MakeExamples(t Truth, n int) ([]mat.Vector, []mat.Vector) {
	var xs, ys []mat.Vector
	inputCardinality, outputCardinality := t.Dims()
	for i := 0; i < n; i++ {
		x := mat.NewVecDense(inputCardinality, nil)
		y := mat.NewVecDense(outputCardinality, nil)
		xs = append(xs, x)
		ys = append(ys, y)

		t.Rand(x)
		t.F(y, x)
	}
	return xs, ys
}
