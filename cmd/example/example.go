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
	K     = 16
	KLog2 = 4
)

var (
	logUpdatePeriod = flag.Duration("log_update_period", 30*time.Second, "time between update")
	batchSize       = flag.Int("batch_size", 128, "batch size")
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
			layers.MakePerceptronLayer(K, KLog2, layers.Sigmoid{}),
			layers.MakePerceptronLayer(KLog2, K, layers.Sigmoid{}),
		},
		LossFunction: lossfunctions.NewSquaredError(),
	}
	alpha := 0.05
	n := int(1e4)
	xs, yHats := mkExamples(n)
	batches := n / *batchSize
	if n%*batchSize != 0 {
		batches++
	}
	lastLogUpdate := time.Now()
	for i := 0; i < int(5e6); i++ {
		start := (i % batches) * *batchSize
		end := start + *batchSize
		if end > n-1 {
			end = n - 1
		}
		e := fn.Train(model, xs[start:end], yHats[start:end], alpha)
		if time.Since(lastLogUpdate) < *logUpdatePeriod {
			continue
		}
		lastLogUpdate = time.Now()
		log.Printf("error: %v\n", e)
		if e < 1e-5 {
			break
		}
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
