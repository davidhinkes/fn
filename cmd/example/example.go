package main

import (
	"fn"
	"fn/layers"
	"fn/lossfunctions"

	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"

	_ "net/http/pprof"
)

const (
	K     = 32
	KLog2 = 5
)

var (
	logUpdatePeriod  = flag.Duration("log_update_period", 1*time.Second, "time between update")
	maxTime          = flag.Duration("max_time", 3*time.Minute, "how long should we crunch on the data?")
	batchSize        = flag.Int("batch_size", 128, "batch size")
	trainingExamples = flag.Int("training_examples", 1024, "")
	alpha            = flag.Float64("alpha", 5e-2, "alpha")
)

func port() string {
	env := os.Getenv("PORT")
	if env == "" {
		return "8080"
	}
	return env
}

func main() {
	flag.Parse()
	go func() {
		log.Println(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%s", port()), nil))
	}()
	model := fn.Model{
		Layers: []fn.Layer{
			layers.MakePerceptronLayer(K, KLog2), layers.MakeBiasLayer(KLog2), layers.Sigmoid{},
			layers.MakePerceptronLayer(KLog2, K), layers.MakeBiasLayer(K),
		},
	}
	lossFunction := lossfunctions.NewSquaredError()
	n := *trainingExamples
	xs, yHats := mkExamples(n)
	batches := n / *batchSize
	if n%*batchSize != 0 {
		batches++
	}
	lastLogUpdateTime := time.Now()
	lastLogUpdateIteration := 0
	startTime := lastLogUpdateTime
	for i := 0; ; i++ {
		start := (i % batches) * *batchSize
		end := start + *batchSize
		if end > n-1 {
			end = n - 1
		}
		e := fn.Train(model, xs[start:end], yHats[start:end], lossFunction, *alpha)
		if time.Since(startTime) > *maxTime {
			break
		}
		deltaTime := time.Since(lastLogUpdateTime)
		if deltaTime < *logUpdatePeriod {
			continue
		}
		lastLogUpdateTime = time.Now()
		iterations := i - lastLogUpdateIteration
		lastLogUpdateIteration = i
		fmt.Printf("loss: %e  iterations: %v\t\t\r", e, iterations)
	}
	tests, _ := mkExamples(10)
	for _, t := range tests {
		y, _ := model.Eval(t)
		log.Printf("%v\n->%v\n\n", mat.Formatted(t), mat.Formatted(y))
	}
}

func mkExamples(n int) ([]mat.Vector, []mat.Vector) {
	var xs []mat.Vector
	var yHats []mat.Vector
	for i := 0; i < n; i++ {
		x := make([]float64, K)
		// x is a one-hot vector
		x[int(rand.Uint32()%uint32(K))] = 1
		xs = append(xs, mat.NewVecDense(K, x))
		// autoencoder
		yHats = append(yHats, mat.NewVecDense(K, x))
	}
	return xs, yHats
}

func random(n int) []float64 {
	ret := make([]float64, n)
	for i := range ret {
		ret[i] = rand.Float64()
	}
	return ret
}
