package fn

import (
	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Layers       []Layer
	LossFunction LossFunction
}

func (m Model) Eval(x mat.Vector) (mat.Vector, []mat.Vector) {
	var upsilons []mat.Vector
	previousUpsilon := x
	for _, l := range m.Layers {
		u := l.F(previousUpsilon)
		upsilons = append(upsilons, u)
		previousUpsilon = u
	}
	return previousUpsilon, upsilons
}
