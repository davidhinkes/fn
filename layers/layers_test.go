package layers

import (
	"testing"

	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"

	"fn"
	"fn/lossfunctions"
	"fn/test"
)

func testLayer(t *testing.T, mkModel func(int) fn.Model, truth test.Truth) {
	t.Helper()
	n, _ := truth.Dims()
	model := mkModel(n)
	startTime := time.Now()
	const (
		batchSize       = 8
		alpha           = 5e-2
		durationSeconds = 15
		allowableError  = 1e-5
	)
	var e float64
	lossFunction := lossfunctions.NewSquaredError()
	xs, ys := test.MakeExamples(truth, batchSize)
	for time.Since(startTime) < durationSeconds*time.Second {
		e,_ = model.Train(xs, ys, lossFunction, alpha)
		if e < allowableError {
			return
		}
	}
	t.Errorf("%s error=%v is too high", t.Name(), e)
}

func TestBiasLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeBiasLayer(n))
	}, identity{N: 64})
}

func TestBiasSerialLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakePerceptronLayer(n, n), MakeBiasLayer(n))
	}, identity{N: 32})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeBiasLayer(n), MakePerceptronLayer(n, n))
	}, identity{N: 16})
}

func TestPerceptronLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakePerceptronLayer(n, n))
	}, identity{N: 32})
}

func TestPerceptronSerialLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		// Use simple serial layers: 16 -> 16 -> 16
		return fn.MakeModel(MakePerceptronLayer(n, n), MakePerceptronLayer(n, n))
	}, identity{N: 16})
}

func TestScalarLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeScalarLayer(n))
	}, identity{N: 64})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeScalarLayer(n), MakeBiasLayer(n))
	}, identity{N: 128})
}

func TestScalarSerialLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakePerceptronLayer(n, n), MakeScalarLayer(n))
	}, identity{N: 16})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeScalarLayer(n), MakePerceptronLayer(n, n))
	}, identity{N: 16})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeScalarLayer(n), MakeScalarLayer(n), MakeScalarLayer(n))
	}, identity{N: 16})
}

func TestScalarScalarLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeScalarLayer(n), MakeScalarLayer(n))
	}, identity{N: 64})
}

func TestScalarSerialLayer2(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakeScalarLayer(n), MakePerceptronLayer(n, n))
	}, identity{N: 64})
}

func TestStaticFuncLayer(t *testing.T) {
	f := func(x float64) float64 { return x }
	d := func(x float64) float64 { return 1 }
	// Because perceptron is on the left, the performance is much worse due to large
	// matrix multiplications.
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(MakePerceptronLayer(n, n), staticFunc{f: f, d: d})
	}, identity{N: 64})
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(staticFunc{f: f, d: d}, MakePerceptronLayer(n, n))
	}, identity{N: 64})
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

func testLayerEqual(t *testing.T, n int, layerA fn.Layer, layerB fn.Layer) {
	t.Helper()
	if a, b := layerA.NumWeights(), layerB.NumWeights(); a != b {
		t.Errorf("NumWeights should be the same, got \n%v\n vs \n%v\n", a, b)
	}
	h := mkRandomSlice(layerA.NumWeights())
	x := mkRandomVec(n)
	var a, b mat.VecDense
	layerA.F(&a, x, h)
	layerB.F(&b, x, h)
	if !mat.Equal(&a, &b) {
		t.Errorf("Func F should return the same, got \n%v\n vs \n%v\n", mat.Formatted(&a), mat.Formatted(&b))
	}
	var aDx, aDh, bDx, bDh mat.Dense
	layerA.D(&aDx, &aDh, x, h)
	layerB.D(&bDx, &bDh, x, h)
	if !mat.Equal(&aDx, &bDx) {
		t.Errorf("Expecting Dx matrix should be equal. Got \n%v\n and \n%v\n",
			mat.Formatted(&aDx), mat.Formatted(&bDx))
	}
	if !mat.Equal(&aDh, &bDh) {
		t.Errorf("Expecting Dh matrix should be equal. Got \n%v\n and \n%v\n",
			mat.Formatted(&aDh), mat.Formatted(&bDh))
	}
}

func TestEquivalentLayer(t *testing.T) {
	f := func(x float64) float64 { return 2 * x }
	d := func(x float64) float64 { return 2 }
	n := 128
	//testLayerEqual(t, n,
	//	fn.Serial(MakePerceptronLayer(n, n), staticFunc{f: f, d: d}),
	//	MakePerceptronLayer(n, n))
	testLayerEqual(t, n,
		fn.Serial(MakePerceptronLayer(n, n), staticFunc{f: f, d: d}),
		fn.Serial(staticFunc{f: f, d: d}, MakePerceptronLayer(n, n)))
}

func mkRandomSlice(n int) []float64 {
	ret := make([]float64, n)
	for i := range ret {
		ret[i] = 2*rand.Float64() - 1
	}
	return ret
}

func mkRandomVec(n int) mat.Vector {
	return mat.NewVecDense(n, mkRandomSlice(n))
}
