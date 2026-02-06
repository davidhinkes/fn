package layers

import (
	"testing"

	"fn"
)

func mkModel() fn.Model {
	const (
		K     = 32
		KLog2 = 5
	)

	return fn.MakeModel(K, Serial(
		Perceptron(KLog2), Bias(), Sigmoid(),
		Perceptron(K), Bias(), Sigmoid(), Scalar(),
	))
}

func TestMarshal(t *testing.T) {
	model := mkModel()
	out, err := model.Marshal(nil)
	if err != nil {
		t.Error(err)
	}
	m := mkModel()
	if err := m.Unmarshal(out); err != nil {
		t.Error(err)
	}
	out2, err := m.Marshal(nil)
	if err != nil {
		t.Error(err)
	}
	if string(out) != string(out2) {
		t.Error("marshalled text does not match")
	}
}
