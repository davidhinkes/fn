package fn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func Train(model Model, xs, yHats []mat.Vector, alpha float64) float64 {
	var totalLoss float64
	n := float64(len(xs))
	partials := make([]*mat.VecDense, len(model.Layers))
	for i, x := range xs {
		y, upsilons := model.Eval(x)
		yHat := yHats[i]
		loss, dLoss := model.LossFunction.F(y, yHat)
		totalLoss += loss / n
		for j := len(model.Layers) - 1; j >= 0; j-- {
			input := x
			if j != 0 {
				input = upsilons[j-1]
			}
			dl, p := model.Layers[j].Backpropagate(input, dLoss)
			dLoss = dl
			if pj := partials[j]; pj == nil {
				partials[j] = mat.VecDenseCopyOf(p)
			} else {
				pj.AddVec(pj, p)
			}
		}
	}
	m := rand.Float64() + 0.5
	for i, layer := range model.Layers {
		p := partials[i]
		p.ScaleVec(-alpha*m, p)
		layer.Learn(p)
	}
	return totalLoss
}
