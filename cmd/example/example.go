package main

import (
	"fn"
	"fn/layers"
	"fn/lossfunctions"

	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func main() {
	model := fn.Model{
		//layers.MakeSigmoidLayer(4,4),
		//layers.MakeSigmoidLayer(4,2),
		layers.MakePerceptronLayer(4, 1, layers.Sigmoid{}),
		layers.MakePerceptronLayer(1, 4, layers.Sigmoid{}),
	}
	t := fn.Trainer{
		Alpha: 0.05,
		Model: model,
		Loss:  lossfunctions.NewSquaredError(),
	}
	for i := uint64(0); i < 100000; i++ {
		xs, yHats := mkExamples(100)
		e := t.Train(xs, yHats)
		if i%250 != 0 {
			continue
		}
		fmt.Printf("error: %v\n", e)
		if e < 1e-5 {
			break
		}
	}
	tests, _ := mkExamples(10)
	for _, t := range tests {
		y, _ := fn.Eval(model, t)
		fmt.Printf("%v\n->%v\n\n", mat.Formatted(t), mat.Formatted(y))
	}
}

func mkExamples(n int) ([]mat.Vector, []mat.Vector) {
	var xs []mat.Vector
	var yHats []mat.Vector
	k := 4
	for i := 0; i < n; i++ {
		x := make([]float64, k)
		x[int(rand.Uint32()%uint32(k))] = 1
		xs = append(xs, mat.NewVecDense(k, x))
		yHats = append(yHats, mat.NewVecDense(k, x))
	}
	return xs, yHats
}

func random(n int) []float64 {
	ret := make([]float64, n)
	for i := range ret {
		ret[i] = rand.Float64()
	}
	return ret
}
