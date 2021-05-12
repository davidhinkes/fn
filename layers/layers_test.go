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

func testLayer(t *testing.T, mkModel func(int) fn.Model, truth test.Truth) {
	t.Helper()
	n, _ := truth.Dims()
	model := mkModel(n)
	startTime := time.Now()
	const (
		batchSize       = 2048
		alpha           = 5e-2
		durationSeconds = 15
	)
	var e float64
	lossFunction := lossfunctions.NewSquaredError()
	xs, ys := test.MakeExamples(truth, batchSize)
	for time.Since(startTime) < durationSeconds*time.Second {
		e = model.Train(xs, ys, lossFunction, alpha)
		if e == 0 {
			break
		}
	}
	t.Logf("%s error=%v", t.Name(), e)
	if e > 1e-4 {
		t.Errorf("error rate too high (e=%v)", e)
	}
}

func TestBiasLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model { return fn.MakeModel(MakeBiasLayer(n)) }, identity{N: 64})
}

func TestPerceptronLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model{ return fn.MakeModel(MakePerceptronLayer(n, n)) }, identity{N: 8})
}

func TestScalarLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model{ return fn.MakeModel(MakeScalarLayer(n)) }, identity{N: 64})
}

func TestStaticFuncLayer(t *testing.T) {
	f := func(x float64) float64 { return -2*x }
	d := func(x float64) float64 { return -2 }
	testLayer(t, func(n int) fn.Model{
		return fn.MakeModel(MakePerceptronLayer(n, n), staticFunc{f: f, d: d}) },
		identity{N: 8})
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
