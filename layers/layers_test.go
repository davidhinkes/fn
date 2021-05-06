package layers

import (
	"testing"

	"math/rand"
	"time"

	"fn"
	"fn/lossfunctions"
	"fn/test"
	"gonum.org/v1/gonum/mat"
)

func testLayer(t *testing.T, mkLayer func(int) fn.Layer, truth test.Truth) {
	t.Helper()
	n, _ := truth.Dims()
	model := fn.MakeModel(mkLayer(n))
	startTime := time.Now()
	const (
		batchSize       = 32
		alpha           = 5e-2
		durationSeconds = 20
	)
	var e float64
	lossFunction := lossfunctions.NewSquaredError()
	for time.Since(startTime) < durationSeconds*time.Second {
		xs, ys := test.MakeExamples(truth, batchSize)
		e = fn.Train(model, xs, ys, lossFunction, alpha)
	}
	t.Logf("%s error=%v", t.Name(), e)
	if e > 1e-5 {
		t.Errorf("error rate too high (e=%v)", e)
	}
}

func TestBiasLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Layer { return MakeBiasLayer(n) }, identity{N: 64})
}

func TestPerceptronLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Layer { return MakePerceptronLayer(n, n) }, identity{N: 64})
}

func TestScalarLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Layer { return MakeScalarLayer(n) }, identity{N: 64})
}

type identity struct {
	N int
}

func (i identity) Dims() (int, int) {
	return i.N, i.N
}

func (i identity) F(dst *mat.VecDense, x mat.Vector) {
	dst.CopyVec(x)
}

func (i identity) Rand(dst *mat.VecDense) {
	for k := 0; k < dst.Len(); k++ {
		dst.SetVec(k, 2*rand.Float64()-1)
	}
}
