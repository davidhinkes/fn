package layers

import (
	"testing"

	"fn/test"
)

func TestPerceptronGradients(t *testing.T) {
	test.GradientCheck(t, Perceptron(3)(4))
}

func TestBiasGradients(t *testing.T) {
	test.GradientCheck(t, Bias()(4))
}

func TestScalarGradients(t *testing.T) {
	test.GradientCheck(t, Scalar()(4))
}

func TestRadialGradients(t *testing.T) {
	test.GradientCheck(t, Radial(3)(4))
}

func TestSigmoidGradients(t *testing.T) {
	test.GradientCheck(t, Sigmoid()(4))
}

func TestReluGradients(t *testing.T) {
	test.GradientCheck(t, Relu()(4))
}

func TestSoftmaxGradients(t *testing.T) {
	test.GradientCheck(t, Softmax()(4))
}
