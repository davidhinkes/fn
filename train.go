package fn

import (
	"log"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

func (model *Model) Train(xs, yHats []mat.Vector, lossFunction LossFunction, alpha float64) float64 {
	// c is the main channel for doing work. For each xs spawn a gothread
	// that will push one item.
	type tuple struct {
		v mat.Vector
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
			y := model.Eval(x)
			loss, dLossDyT := lossFunction.F(y, yHat)
			_, dYdW := model.layer.D(x, model.weights)
			// dLdWT = (dLdY * dYdW)T
			dLdWT := mulVec(mat.Transpose{Matrix: dYdW}, dLossDyT)
			c <- tuple{
				l: loss,
				v: dLdWT,
			}
		}(x, yHats[i])
	}
	var meanLoss float64
	dLossdWT := mat.NewVecDense(model.layer.NumWeights(), nil)
	for p := range c {
		meanLoss += p.l / float64(n)
		dLossdWT.AddScaledVec(dLossdWT, 1./float64(n), p.v)
	}
	if alpha == 0 || meanLoss == 0 {
		// if alpha is zero, we don't want any learning
		// if meanLoss is zero, there is nothing to learn
		return meanLoss
	}
	//sum := mat.Dot(dLossdWT, dLossdWT)
	w := mat.NewVecDense(len(model.weights), model.weights)
	//w.AddScaledVec(w, -alpha*meanLoss/sum, dLossdWT)
	w.AddScaledVec(w, -alpha, dLossdWT)
	return meanLoss
}

func mulVec(m mat.Matrix, v mat.Vector) mat.Vector {
	var ret mat.VecDense
	ret.MulVec(m, v)
	return &ret
}

type TrainOptions struct {
	Alpha          float64
	BatchSize      int
	LossFunction   LossFunction
	TrainDuration  time.Duration
	StatusDuration time.Duration
}

// TrainBatch
// This is not a funtion of Model to convey the user shouldn't be using m while this is running.
// An alternative idea is to have the user provide a function callback. IMO, use of channels is cleaner.
func (m *Model) TrainBatch(xs, ys []mat.Vector, opts TrainOptions, f func(int, float64)) float64 {
	if a, b := len(xs), len(ys); a != b {
		log.Fatalf("expecting sizes of xs & yHats to be equal; got %v, %v", a, b)
	}
	lastStatusCallTime := time.Now()
	lastStatusCallIteration := 0
	startTime := lastStatusCallTime
	var e float64
	for i := 0; time.Since(startTime) < opts.TrainDuration; i++ {
		bxs := batch(xs, opts.BatchSize, i)
		bys := batch(ys, opts.BatchSize, i)

		e = m.Train(bxs, bys, opts.LossFunction, opts.Alpha)
		if time.Since(lastStatusCallTime) < opts.StatusDuration {
			continue
		}
		f(i-lastStatusCallIteration, e)
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
