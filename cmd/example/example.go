package main

import (
	"fn"
	"fn/layers"
	"fn/lossfunctions"
	"fn/test"

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
		layers.MakePerceptronLayer(K, KLog2), layers.MakeBiasLayer(KLog2), layers.MakeSigmoid(),
		layers.MakePerceptronLayer(KLog2, K), layers.MakeBiasLayer(K), layers.MakeSigmoid(), layers.MakeScalarLayer(K),
	)
	lossFunction := lossfunctions.NewSquaredError()
	truth := oneHot{Cardinality: K}
	vxs, vys := test.MakeExamples(truth, *trainingExamples)
	lastLogUpdateTime := time.Now()
	lastLogUpdateIteration := 0
	startTime := lastLogUpdateTime
	for i := 0; ; i++ {
		xs, ys := test.MakeExamples(truth, *batchSize)
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
	tests, _ := test.MakeExamples(truth, *trainingExamples)
	for _, t := range tests {
		y, _ := model.Eval(t)
		log.Printf("%v\n->%v\n\n", mat.Formatted(t), mat.Formatted(y))
	}
}

type oneHot struct {
	Cardinality int
}

var _ test.Truth = oneHot{}

func (o oneHot) Dims() (int, int) {
	return o.Cardinality, o.Cardinality
}

func (o oneHot) F(dst *mat.VecDense, x mat.Vector) {
	dst.CloneFromVec(x)
}

func (o oneHot) Rand(dst *mat.VecDense) {
	dst.Zero()
	dst.SetVec(rand.Int()%o.Cardinality, 1.0)
}
