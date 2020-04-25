package main

import (
	"fn"
	"fn/layers"
	"fn/lossfunctions"

	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

const (
	K = 16
	KLog2 = 4
)

func main() {
	model := fn.Model{
		layers.MakePerceptronLayer(K, KLog2, layers.Sigmoid{}),
		layers.MakePerceptronLayer(KLog2, K, layers.Sigmoid{}),
	}
	t := fn.Trainer{
		Alpha: 0.05,
		Model: model,
		Loss:  lossfunctions.NewSquaredError(),
	}
	n := int(1e4)
	xs, yHats := mkExamples(n)
	batchSize := 128
	batches := n/batchSize
	if n % batchSize != 0 {
		batches++
	}
	for i := 0; i < int(5e6); i++ {
		start := (i % batches) * batchSize
		end := start + batchSize
		if end > n - 1 {
		  end = n - 1
		}
		e := t.Train(xs[start:end], yHats[start:end])
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
		y, _ := model.Eval(t)
		fmt.Printf("%v\n->%v\n\n", mat.Formatted(t), mat.Formatted(y))
	}
}

func mkExamples(n int) ([]mat.Vector, []mat.Vector) {
	var xs []mat.Vector
	var yHats []mat.Vector
	for i := 0; i < n; i++ {
		x := make([]float64, K)
		x[int(rand.Uint32()%uint32(K))] = 1
		xs = append(xs, mat.NewVecDense(K, x))
		yHats = append(yHats, mat.NewVecDense(K, x))
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
