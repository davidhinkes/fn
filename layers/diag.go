package layers

import (
	"gonum.org/v1/gonum/mat"
)

// diag creates a diagonal matrix from a vector.
//
// IMPORTANT: We use a Dense matrix, not DiagDense, despite the memory overhead.
// This is because gonum's Dense.Mul() method has extremely poor performance when
// multiplying DiagDense × Dense (450x+ slower than Dense × Dense). Since these
// diagonal matrices are frequently used in backpropagation (e.g., in staticFunc
// layers for activation function derivatives), using DiagDense causes catastrophic
// performance degradation in training loops.
//
// Performance comparison for 64×64 matrices:
//   - DiagDense × Dense: ~100ms per multiplication
//   - Dense × Dense: ~0.2ms per multiplication
//
// Memory trade-off: Dense uses n² space vs DiagDense's n space, but the performance
// gain (450x+) far outweighs the memory cost for typical neural network dimensions.
func diag(dst *mat.Dense, x mat.Vector) {
	n := x.Len()
	dst.Zero()
	for i := 0; i < n; i++ {
		dst.Set(i, i, x.AtVec(i))
	}
}

// diagFromSlice creates a diagonal matrix from a slice.
//
// See diag() for why we use Dense instead of DiagDense.
func diagFromSlice(dst *mat.Dense, x []float64) {
	n := len(x)
	dst.Zero()
	for i := 0; i < n; i++ {
		dst.Set(i, i, x[i])
	}
}
