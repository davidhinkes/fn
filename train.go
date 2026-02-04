package fn

import (
	"fn/matrixpool"
	"log"
	"time"

	"gonum.org/v1/gonum/mat"
)

func (model *Model) Train(xs, yHats []mat.Vector, lossFunction LossFunction, alpha float64) (float64, float64) {
	inputs, outputs, numWeights := model.layer.Shape()
	n := len(xs)

	y := matrixpool.GetVec(outputs)
	defer matrixpool.PutVec(y)
	dLdY := matrixpool.GetVec(outputs)
	defer matrixpool.PutVec(dLdY)
	dLdX := matrixpool.GetVec(inputs)
	defer matrixpool.PutVec(dLdX)
	dLdWi := matrixpool.GetVec(numWeights)
	defer matrixpool.PutVec(dLdWi)
	dLdW := matrixpool.GetVec(numWeights)
	defer matrixpool.PutVec(dLdW)
	dLdW.Zero()

	var loss float64
	for i, x := range xs {
		model.Eval(y, x)
		loss += lossFunction.F(dLdY, y, yHats[i])
		model.layer.D(dLdX, dLdWi, dLdY, x, model.weights)
		dLdW.AddVec(dLdW, dLdWi)
	}

	w := mat.NewVecDense(len(model.weights), model.weights)
	dLdW.ScaleVec(1./float64(n), dLdW)
	meanLoss := loss / float64(n)
	gradientNorm := mat.Norm(dLdW, 2)
	if alpha == 0 || meanLoss == 0 {
		return meanLoss, gradientNorm
	}
	w.AddScaledVec(w, -alpha, dLdW)
	return meanLoss, gradientNorm
}

type TrainOptions struct {
	Alpha          float64
	BatchSize      int
	LossFunction   LossFunction
	TrainDuration  time.Duration
	StatusDuration time.Duration
}

// TrainBatch
// This is not a function of Model to convey the user shouldn't be using m while this is running.
// An alternative idea is to have the user provide a function callback. IMO, use of channels is cleaner.
func (m *Model) TrainBatch(xs, ys []mat.Vector, opts TrainOptions, f func(int, float64, float64)) float64 {
	if a, b := len(xs), len(ys); a != b {
		log.Fatalf("expecting sizes of xs & yHats to be equal; got %v, %v", a, b)
	}
	lastStatusCallTime := time.Now()
	lastStatusCallIteration := 0
	startTime := lastStatusCallTime
	var e float64
	var gradientNorm float64
	for i := 0; time.Since(startTime) < opts.TrainDuration; i++ {
		bxs := batch(xs, opts.BatchSize, i)
		bys := batch(ys, opts.BatchSize, i)

		e, gradientNorm = m.Train(bxs, bys, opts.LossFunction, opts.Alpha)
		if time.Since(lastStatusCallTime) < opts.StatusDuration {
			continue
		}
		f(i-lastStatusCallIteration, e, gradientNorm)
		lastStatusCallIteration = i
		lastStatusCallTime = time.Now()
	}
	return e
}

func batch(x []mat.Vector, batchSize int, i int) []mat.Vector {
	numBatches := len(x) / batchSize
	if len(x)%batchSize != 0 {
		numBatches++
	}
	start := (i % numBatches) * batchSize
	end := start + batchSize
	if end > len(x) {
		end = len(x)
	}
	return x[start:end]
}
