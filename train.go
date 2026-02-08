package fn

import (
	"fn/matrixpool"
	"log"
	"time"

	"gonum.org/v1/gonum/mat"
)

func (model *Model) Train(X, YHat *mat.Dense, lossFunction LossFunction, alpha float64) (float64, float64) {
	inputs, outputs, numWeights := model.layer.Shape()
	b, _ := X.Dims()

	Y := matrixpool.GetDense(b, outputs)
	defer matrixpool.PutDense(Y)
	dLdY := matrixpool.GetDense(b, outputs)
	defer matrixpool.PutDense(dLdY)
	dLdX := matrixpool.GetDense(b, inputs)
	defer matrixpool.PutDense(dLdX)
	dLdW := matrixpool.GetVec(numWeights)
	defer matrixpool.PutVec(dLdW)

	// Forward pass — single GEMM per layer.
	model.layer.F(Y, X, model.weights)

	// Compute loss per example (LossFunction is still per-vector).
	var loss float64
	for row := 0; row < b; row++ {
		loss += lossFunction.F(dLdY.RowView(row).(*mat.VecDense), Y.RowView(row), YHat.RowView(row))
	}

	// Backward pass — single call, dLdW comes back summed over batch.
	model.layer.D(dLdX, dLdW, dLdY, X, model.weights)

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
