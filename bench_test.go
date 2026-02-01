package fn_test

import (
	"fmt"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"

	"fn"
	"fn/layers"
	"fn/lossfunctions"
	"fn/test"
)

type benchIdentity struct {
	n   int
	rng *rand.Rand
}

func (id benchIdentity) Dims() (int, int) { return id.n, id.n }

func (id benchIdentity) F(dst *mat.VecDense, x mat.Vector) {
	dst.CopyVec(x)
}

func (id benchIdentity) Rand(dst *mat.VecDense) {
	for k := 0; k < dst.Len(); k++ {
		dst.SetVec(k, 2*id.rng.Float64()-1)
	}
}

var _ test.Truth = benchIdentity{}

func BenchmarkTrain(b *testing.B) {
	dims := []int{8, 16, 32, 64}
	for _, dim := range dims {
		b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			rand.Seed(42)

			model := fn.MakeModel(dim,
				layers.Perceptron(dim), layers.Bias(), layers.Sigmoid(),
				layers.Perceptron(dim), layers.Bias(),
			)

			truth := benchIdentity{n: dim, rng: rng}
			xs, ys := test.MakeExamples(truth, 8)
			lossFunc := lossfunctions.NewSquaredError()

			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				model.Train(xs, ys, lossFunc, 0.05)
			}
		})
	}
}
