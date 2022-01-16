package main

import (
	"fn"
	"fn/layers"
	"fn/lossfunctions"
	"fn/test"

	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"time"

	"gonum.org/v1/gonum/mat"

	"cloud.google.com/go/storage"

	_ "net/http/pprof"
)

const (
	K = 1
)

var (
	logUpdatePeriod  = flag.Duration("log_update_period", 1*time.Second, "time between update")
	maxTime          = flag.Duration("max_time", 3*time.Minute, "how long should we crunch on the data?")
	batchSize        = flag.Int("batch_size", 128, "batch size")
	trainingExamples = flag.Int("training_examples", 1024, "")
	alpha            = flag.Float64("alpha", 5e-2, "alpha")
	storageURI       = flag.String("storage_uri", "", "")
	gsRegexp         = regexp.MustCompile(`^gs:\/\/([^\/]*)\/(.*)$`)
)

func port() string {
	env := os.Getenv("PORT")
	if env == "" {
		return "8080"
	}
	return env
}

func writeStorage(ctx context.Context, uri string, blob []byte) error {
	if uri == "" {
		return nil
	}
	parts := gsRegexp.FindStringSubmatch(uri)
	if len(parts) != 3 {
		return fmt.Errorf("gs uri decode fail: %s, parts: %v", uri, parts)
	}
	bucketName, objectName := parts[1], parts[2]
	c, err := storage.NewClient(ctx)
	if err != nil {
		return err
	}
	defer c.Close()
	w := c.Bucket(bucketName).Object(objectName).NewWriter(ctx)
	defer w.Close()
	_, err = w.Write(blob)
	return err
}

func main() {
	flag.Parse()
	ctx := context.Background()
	go func() {
		log.Println(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%s", port()), nil))
	}()
	truth := euclidian{}
	inputCardinality, outputCardinality := truth.Dims()
	model := fn.MakeModel(
		layers.MakeRadialLayer(inputCardinality, K),
		layers.MakePerceptronLayer(K, outputCardinality), layers.MakeBiasLayer(outputCardinality),
	)
	lossFunction := lossfunctions.NewSquaredError()
	vxs, vys := test.MakeExamples(truth, *trainingExamples)
	lastLogUpdateTime := time.Now()
	lastLogUpdateIteration := 0
	startTime := lastLogUpdateTime
	for i := 0; ; i++ {
		xs, ys := test.MakeExamples(truth, *batchSize)
		e := model.Train(xs, ys, lossFunction, *alpha)
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
		e = model.Train(vxs, vys, lossFunction, 0)
		log.Printf("loss: %e  iterations: %v", e, iterations)
	}
	tests, _ := test.MakeExamples(truth, *trainingExamples)
	for _, t := range tests {
		y := model.Eval(t)
		log.Printf("%v\n->%v\n\n", mat.Formatted(t), mat.Formatted(y))
	}

	blob, err := model.Marshal(tests[0])
	if err != nil {
		log.Fatal(err)
	}
	if err := writeStorage(ctx, *storageURI, blob); err != nil {
		log.Fatal(err)
	}
}

type euclidian struct {
}

func (e euclidian) Dims() (int, int) {
	return 2, 1
}

func (e euclidian) F(dst *mat.VecDense, x mat.Vector) {
	dst.SetVec(0, math.Sqrt(mat.Dot(x, x)))
}

func (e euclidian) Rand(dst *mat.VecDense) {
	dst.Zero()
	n, _ := e.Dims()
	for i := 0; i < n; i++ {
		dst.SetVec(i, 1000*(rand.Float64()-0.5))
	}
}

var _ test.Truth = euclidian{}
