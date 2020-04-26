package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Trainer struct {
	Alpha float64
	Model Model
}

func (t Trainer) Train(xs, yHats []mat.Vector) float64 {
	var totalLoss float64
	n := float64(len(xs))
	for i, x := range xs {
		y, upsilons := t.Model.Eval(x)
		yHat := yHats[i]
		loss, dLoss := t.Model.LossFunction.F(y, yHat)
		totalLoss += loss / n
		for j := len(t.Model.Layers) - 1; j >= 0; j-- {
			input := x
			if j != 0 {
				input = upsilons[j-1]
			}
			dLoss = t.Model.Layers[j].Learn(input, dLoss, t.Alpha)
		}
	}
	return totalLoss
}
