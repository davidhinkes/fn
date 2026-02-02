package fn

import (
	"fn/matrixpool"
	"log"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

func (model *Model) Train(xs, yHats []mat.Vector, lossFunction LossFunction, alpha float64) (float64, float64) {
	// c is the main channel for doing work. For each xs spawn a goroutine
	// that will push one item.
	type tuple struct {
		v *mat.VecDense
		l float64
	}
	c := make(chan tuple)
	wg := sync.WaitGroup{}
	n := len(xs)
	wg.Add(n)
	// Close c only after all workers have completed.
	go func() {
		wg.Wait()
		close(c)
	}()
	for i, x := range xs {
		go func(x mat.Vector, yHat mat.Vector) {
			defer wg.Done()
			inputs, outputs, weights := model.layer.Shape()
			y := matrixpool.GetVec(outputs)
			defer matrixpool.PutVec(y)
			model.Eval(y, x)
			dLossDyT := matrixpool.GetVec(outputs)
			defer matrixpool.PutVec(dLossDyT)
			loss := lossFunction.F(dLossDyT, y, yHat)
			dYdX := matrixpool.GetDense(outputs, inputs)
			defer matrixpool.PutDense(dYdX)
			dYdW := matrixpool.GetDense(outputs, weights)
			defer matrixpool.PutDense(dYdW)
			model.layer.D(dYdX, dYdW, x, model.weights)
			// dLdWT = (dLdY * dYdW)T
			dLdWT := matrixpool.GetVec(weights) // we reclaim this later in the other thread!
			dLdWT.MulVec(mat.Transpose{Matrix: dYdW}, dLossDyT)
			c <- tuple{
				l: loss,
				v: dLdWT,
			}
		}(x, yHats[i])
	}
	var loss float64
	_, _, numWeights := model.layer.Shape()
	dLossdWT := matrixpool.GetVec(numWeights)
	defer matrixpool.PutVec(dLossdWT)
	for p := range c {
		loss += p.l
		dLossdWT.AddVec(dLossdWT, p.v)
		matrixpool.PutVec(p.v)
	}
	w := mat.NewVecDense(len(model.weights), model.weights)
	dLossdWT.ScaleVec(1./float64(n), dLossdWT)
	meanLoss := loss / float64(n)
	gradientNorm := mat.Norm(dLossdWT, 2)
	if alpha == 0 || meanLoss == 0 {
		// if alpha is zero, we don't want any learning
		// if loss is zero, there is nothing to learn
		return meanLoss, gradientNorm
	}
	w.AddScaledVec(w, -alpha, dLossdWT)
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
