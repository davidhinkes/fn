package fn

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

type partialDerivative struct {
	v []mat.Vector
	l float64
}

func Train(model Model, xs, yHats []mat.Vector, lossFunction LossFunction, alpha float64) float64 {
	// c is the main channel for doing work. For each xs spawn a gothread
	// that will push one item.
	c := make(chan partialDerivative)
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
			y, upsilons := model.Eval(x)
			loss, dLossDyT := lossFunction.F(y, yHat)
			partial := partialDerivative{
				l: loss,
				v: make([]mat.Vector, len(model.nodes)),
			}
			for j := len(model.nodes) - 1; j >= 0; j-- {
				input := x
				if j != 0 {
					input = upsilons[j-1]
				}
				hyperparameters := model.nodes[j].hyperparameters
				dYdX, dYdH := model.nodes[j].layer.D(input, hyperparameters)
				// dYdH being nil is valid, meaning the Zero matrix
				if dYdH == nil {
					partial.v[j] = nil
				} else {
					partial.v[j] = mulVec(mat.Transpose{dYdH}, dLossDyT)
				}
				dLossDyT = mulVec(mat.Transpose{dYdX}, dLossDyT)
			}
			c <- partial
		}(x, yHats[i])
	}
	var meanLoss float64
	partials := make([]*mat.VecDense, len(model.nodes))
	for p := range c {
		meanLoss += p.l / float64(n)
		agg(partials, p.v, n)
	}
	if alpha == 0 {
		return meanLoss 
	}
	var sum float64
	for _, p := range partials {
		if p == nil {
			continue
		}
		sum += mat.Dot(p, p)
	}
	for i, node := range model.nodes {
		if partials[i] == nil {
			continue
		}
		hSlice := node.hyperparameters
		h := mat.NewVecDense(len(hSlice), hSlice)
		p := partials[i]
		h.AddScaledVec(h, -alpha*meanLoss/sum, p)
	}
	return meanLoss 
}

func mulVec(m mat.Matrix, v mat.Vector) mat.Vector {
	var ret mat.VecDense
	ret.MulVec(m, v)
	return &ret
}

func agg(ps []*mat.VecDense, vs []mat.Vector, n int) {
	for i, p := range ps {
		v := vs[i]
		if v == nil {
			continue
		}
		if p == nil {
			ps[i] = mat.NewVecDense(v.Len(), nil)
			p = ps[i]
		}
		p.AddScaledVec(p, 1./float64(n), v)
	}
}
