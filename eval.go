package fn

import (
	"gonum.org/v1/gonum/mat"
)

func Eval(model Model, x mat.Vector) (mat.Vector, []mat.Vector) {
	var upsilons []mat.Vector
	previousUpsilon := x
	for _, l := range []Layer(model) {
		u := l.F(previousUpsilon)
		upsilons = append(upsilons, u)
		previousUpsilon = u
	}
	return previousUpsilon, upsilons
}
