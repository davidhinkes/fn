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

	return fn.MakeModel(
		MakePerceptronLayer(K, KLog2), MakeBiasLayer(KLog2), MakeSigmoid(),
		MakePerceptronLayer(KLog2, K), MakeBiasLayer(K), MakeSigmoid(), MakeScalarLayer(K),
	)
}

func TestMarshal(t *testing.T) {
	out, err := mkModel().Marshal()
	if err != nil {
		t.Error(err)
	}
	m := mkModel()
	if err := m.Unmarshal(out); err != nil {
		t.Error(err)
	}
	out2, err := m.Marshal()
	if err != nil {
		t.Error(err)
	}
	if string(out) != string(out2) {
		t.Error("marshalled text does not match")
	}
}
