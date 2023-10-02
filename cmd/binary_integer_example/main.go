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
	K     = 32
	KLog2 = 5
)

var (
	logUpdatePeriod  = flag.Duration("log_update_period", 1*time.Second, "time between update")
	maxTime          = flag.Duration("max_time", 3*time.Minute, "how long should we crunch on the data?")
	batchSize        = flag.Int("batch_size", 32, "batch size")
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
	model := fn.MakeModel(
		layers.MakePerceptronLayer(K, KLog2), layers.MakeBiasLayer(KLog2), layers.MakeSigmoid(),
		layers.MakePerceptronLayer(KLog2, K), layers.MakeBiasLayer(K), layers.MakeSigmoid(),
	)
	opts := fn.TrainOptions{
		Alpha: *alpha,
		TrainDuration: *maxTime,
		BatchSize: *batchSize,
		LossFunction: lossfunctions.NewSquaredError(),
		StatusDuration: *logUpdatePeriod,
	}
	truth := oneHot{Cardinality: K}
	xs, ys := test.MakeExamples(truth, *trainingExamples)
	model.TrainBatch(xs, ys, opts, func(i int, e float64){
		log.Printf("loss: %e  iterations: %v", e, i)
	})

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
