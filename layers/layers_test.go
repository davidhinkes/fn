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
		e, _ = model.Train(xs, ys, lossFunction, alpha)
		if e < allowableError {
			return
		}
	}
	t.Errorf("%s error=%v is too high", t.Name(), e)
}

func TestBiasLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Bias())
	}, identity{N: 64})
}

func TestBiasSerialLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Perceptron(n), Bias()))
	}, identity{N: 32})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Bias(), Perceptron(n)))
	}, identity{N: 16})
}

func TestPerceptronLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Perceptron(n))
	}, identity{N: 32})
}

func TestPerceptronSerialLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		// Use simple serial layers: 16 -> 16 -> 16
		return fn.MakeModel(n, Serial(Perceptron(n), Perceptron(n)))
	}, identity{N: 16})
}

func TestScalarLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Scalar())
	}, identity{N: 64})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Scalar(), Bias()))
	}, identity{N: 128})
}

func TestScalarSerialLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Perceptron(n), Scalar()))
	}, identity{N: 16})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Scalar(), Perceptron(n)))
	}, identity{N: 16})

	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Scalar(), Scalar(), Scalar()))
	}, identity{N: 16})
}

func TestScalarScalarLayer(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Scalar(), Scalar()))
	}, identity{N: 64})
}

func TestScalarSerialLayer2(t *testing.T) {
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Scalar(), Perceptron(n)))
	}, identity{N: 64})
}

func TestStaticFuncLayer(t *testing.T) {
	f := func(x float64) float64 { return x }
	d := func(x float64) float64 { return 1 }
	idBuilder := func(inputs int) fn.Layer {
		return staticFunc{f: f, d: d, n: inputs}
	}
	// Because perceptron is on the left, the performance is much worse due to large
	// matrix multiplications.
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(Perceptron(n), idBuilder))
	}, identity{N: 64})
	testLayer(t, func(n int) fn.Model {
		return fn.MakeModel(n, Serial(idBuilder, Perceptron(n)))
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
	_, _, a := layerA.Shape()
	_, _, b := layerB.Shape()
	if a != b {
		t.Errorf("weights should be the same, got \n%v\n vs \n%v\n", a, b)
	}
	h := mkRandomSlice(a)
	x := mkRandomVec(n)
	aInputs, aOutputs, aWeights := layerA.Shape()
	bInputs, bOutputs, bWeights := layerB.Shape()
	aVec := mat.NewVecDense(aOutputs, nil)
	bVec := mat.NewVecDense(bOutputs, nil)
	layerA.F(aVec, x, h)
	layerB.F(bVec, x, h)
	if !mat.Equal(aVec, bVec) {
		t.Errorf("Func F should return the same, got \n%v\n vs \n%v\n", mat.Formatted(aVec), mat.Formatted(bVec))
	}
	// Test backward pass with random upstream gradient
	dLdY := mkRandomVec(aOutputs).(*mat.VecDense)
	aDx := mat.NewVecDense(aInputs, nil)
	aDh := mat.NewVecDense(aWeights, nil)
	bDx := mat.NewVecDense(bInputs, nil)
	bDh := mat.NewVecDense(bWeights, nil)
	layerA.D(aDx, aDh, dLdY, x, h)
	layerB.D(bDx, bDh, dLdY, x, h)
	if !mat.EqualApprox(aDx, bDx, 1e-10) {
		t.Errorf("Expecting dLdX should be equal. Got \n%v\n and \n%v\n",
			mat.Formatted(aDx), mat.Formatted(bDx))
	}
	if !mat.EqualApprox(aDh, bDh, 1e-10) {
		t.Errorf("Expecting dLdH should be equal. Got \n%v\n and \n%v\n",
			mat.Formatted(aDh), mat.Formatted(bDh))
	}
}

func TestEquivalentLayer(t *testing.T) {
	f := func(x float64) float64 { return 2 * x }
	d := func(x float64) float64 { return 2 }
	n := 128

	doubleBuilder := func(inputs int) fn.Layer {
		return staticFunc{f: f, d: d, n: inputs}
	}

	testLayerEqual(t, n,
		Serial(Perceptron(n), doubleBuilder)(n),
		Serial(doubleBuilder, Perceptron(n))(n))
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
