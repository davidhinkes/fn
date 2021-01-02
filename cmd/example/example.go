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
	model := fn.MakeModel(
		layers.MakePerceptronLayer(K, KLog2), layers.MakeBiasLayer(KLog2), layers.MakeRelu(),
		layers.MakePerceptronLayer(KLog2, K), layers.MakeBiasLayer(K), layers.MakeRelu(),
	)
	lossFunction := lossfunctions.NewSquaredError()
	vxs, vys := mkExamples(*trainingExamples)
	lastLogUpdateTime := time.Now()
	lastLogUpdateIteration := 0
	startTime := lastLogUpdateTime
	for i := 0; ; i++ {
		xs, ys := mkExamples(*batchSize)
		e := fn.Train(model, xs, ys, lossFunction, *alpha)
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
		e = fn.Train(model, vxs, vys, lossFunction, 0)
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
