package fn

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

func Train(model Model, xs, yHats []mat.Vector, lossFunction LossFunction, alpha float64) float64 {
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
			dLdWT := mulVec(mat.Transpose{dYdW}, dLossDyT)
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
	sum := mat.Dot(dLossdWT, dLossdWT)
	w := mat.NewVecDense(len(model.weights), model.weights)
	w.AddScaledVec(w, -alpha*meanLoss/sum, dLossdWT)
	return meanLoss
}

func mulVec(m mat.Matrix, v mat.Vector) mat.Vector {
	var ret mat.VecDense
	ret.MulVec(m, v)
	return &ret
}
