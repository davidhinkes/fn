package fn

import (
	"gonum.org/v1/gonum/mat"
)

func Train(model Model, xs, yHats []mat.Vector, alpha float64) float64 {
	var totalLoss float64
	n := float64(len(xs))
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
			dLoss = model.Layers[j].Learn(input, dLoss, alpha)
		}
	}
	return totalLoss
}
