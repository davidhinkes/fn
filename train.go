package fn

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

type partialDerivative struct {
	v []mat.Vector
	l float64
}

func Train(model Model, xs, yHats []mat.Vector, alpha float64) float64 {
	// c is the main channel for doing work. For each xs spawn a gothread
	// that will push one item.
	c := make(chan partialDerivative)
	wg := sync.WaitGroup{}
	wg.Add(len(xs))
	// Close c only after all workers have completed.
	go func() {
		wg.Wait()
		close(c)
	}()
	for i, x := range xs {
		go func(x mat.Vector, yHat mat.Vector) {
			defer wg.Done()
			y, upsilons := model.Eval(x)
			loss, dLossDY := model.LossFunction.F(y, yHat)
			partial := partialDerivative{
				l: loss,
				v: make([]mat.Vector, len(model.Layers)),
			}
			for j := len(model.Layers) - 1; j >= 0; j-- {
				input := x
				if j != 0 {
					input = upsilons[j-1]
				}
				dLossDY, partial.v[j] = model.Layers[j].Backpropagate(input, dLossDY)
			}
			c <- partial
		}(x, yHats[i])
	}
	var totalLoss float64
	partials := make([]*mat.VecDense, len(model.Layers))
	for p := range c {
		totalLoss += p.l
		agg(partials, p.v)
	}
	// Apply the updates to the model
	for i, layer := range model.Layers {
		p := partials[i]
		p.ScaleVec(-alpha, p)
		layer.Learn(p)
	}
	return totalLoss
}

func agg(ps []*mat.VecDense, vs []mat.Vector) {
	for i, p := range ps {
		v := vs[i]
		if p == nil {
			ps[i] = mat.VecDenseCopyOf(v)
			continue
		}
		p.AddVec(p, v)
	}
}
