package fn

import (
	"fn/matrixpool"
	"log"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Train performs one gradient descent step on a batch of examples.
// X is (b × inputs), YHat is (b × outputs) where b is the batch size.
// Returns mean loss and gradient norm.
//
// The forward and backward passes operate on the full batch matrix in a
// single call per layer, enabling level-3 BLAS (GEMM) instead of per-example
// matrix-vector operations. The loss function is still per-vector, so we
// iterate over rows to build dLdY before passing the full gradient matrix
// to the backward pass.
func (model *Model) Train(X, YHat *mat.Dense, lossFunction LossFunction, alpha float64) (float64, float64) {
	inputs, outputs, numWeights := model.layer.Shape()
	b, _ := X.Dims()

	// All temporaries are pooled to avoid allocation in this hot path.
	Y := matrixpool.GetDense(b, outputs)
	defer matrixpool.PutDense(Y)
	dLdY := matrixpool.GetDense(b, outputs)
	defer matrixpool.PutDense(dLdY)
	dLdX := matrixpool.GetDense(b, inputs)
	defer matrixpool.PutDense(dLdX)
	dLdW := matrixpool.GetVec(numWeights)
	defer matrixpool.PutVec(dLdW)

	// Forward pass — batch matrix through all layers.
	model.layer.F(Y, X, model.weights)

	// Compute loss and build the dLdY gradient matrix row by row.
	// LossFunction operates per-vector, so we iterate over rows. We use
	// RowViewOf to reuse VecDense headers rather than RowView, which would
	// allocate a new VecDense on the heap for every row. RowViewOf points
	// the receiver at a row of the matrix without copying or allocating.
	var loss float64
	var yRow, dLdYRow, yHatRow mat.VecDense
	for row := 0; row < b; row++ {
		yRow.RowViewOf(Y, row)
		dLdYRow.RowViewOf(dLdY, row)
		yHatRow.RowViewOf(YHat, row)
		loss += lossFunction.F(&dLdYRow, &yRow, &yHatRow)
	}

	// Backward pass — single call for the full batch. Each layer's D
	// sums weight gradients (dLdW) internally across the batch.
	model.layer.D(dLdX, dLdW, dLdY, X, model.weights)

	// Average gradients and loss over the batch, then update weights.
	w := mat.NewVecDense(len(model.weights), model.weights)
	dLdW.ScaleVec(1./float64(b), dLdW)
	meanLoss := loss / float64(b)
	gradientNorm := mat.Norm(dLdW, 2)
	if alpha == 0 || meanLoss == 0 {
		return meanLoss, gradientNorm
	}
	w.AddScaledVec(w, -alpha, dLdW)
	return meanLoss, gradientNorm
}

type TrainOptions struct {
	Alpha          float64
	BatchSize      int
	LossFunction   LossFunction
	TrainDuration  time.Duration
	StatusDuration time.Duration
}

// TrainBatch trains the model over multiple batches for the configured duration.
func (m *Model) TrainBatch(X, Y *mat.Dense, opts TrainOptions, f func(int, float64, float64)) float64 {
	rows, n := X.Dims()
	_, yc := Y.Dims()
	if yr, _ := Y.Dims(); yr != rows {
		log.Fatalf("expecting sizes of X & Y to be equal; got %v, %v", rows, yr)
	}
	lastStatusCallTime := time.Now()
	lastStatusCallIteration := 0
	startTime := lastStatusCallTime
	var e float64
	var gradientNorm float64
	for i := 0; time.Since(startTime) < opts.TrainDuration; i++ {
		bx := batchDense(X, rows, n, opts.BatchSize, i)
		by := batchDense(Y, rows, yc, opts.BatchSize, i)

		e, gradientNorm = m.Train(bx, by, opts.LossFunction, opts.Alpha)
		if time.Since(lastStatusCallTime) < opts.StatusDuration {
			continue
		}
		f(i-lastStatusCallIteration, e, gradientNorm)
		lastStatusCallIteration = i
		lastStatusCallTime = time.Now()
	}
	return e
}

func batchDense(X *mat.Dense, totalRows, cols, batchSize, i int) *mat.Dense {
	numBatches := totalRows / batchSize
	if totalRows%batchSize != 0 {
		numBatches++
	}
	start := (i % numBatches) * batchSize
	end := start + batchSize
	if end > totalRows {
		end = totalRows
	}
	return X.Slice(start, end, 0, cols).(*mat.Dense)
}
