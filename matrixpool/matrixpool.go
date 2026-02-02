package matrixpool

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

type d struct {
	r int
	c int
}

var (
	matrixPool = map[d]*sync.Pool{}
	vecPool    = map[int]*sync.Pool{}
	mutex      sync.Mutex
)

func getMatrixPool(d d) *sync.Pool {
	mutex.Lock()
	defer mutex.Unlock()
	p, ok := matrixPool[d]
	if ok {
		return p
	}
	pool := sync.Pool{
		New: func() any { return mat.NewDense(d.r, d.c, nil) },
	}
	matrixPool[d] = &pool
	return &pool
}

func getVecPool(n int) *sync.Pool {
	mutex.Lock()
	defer mutex.Unlock()
	p, ok := vecPool[n]
	if ok {
		return p
	}
	pool := sync.Pool{
		New: func() any { return mat.NewVecDense(n, nil) },
	}
	vecPool[n] = &pool
	return &pool
}

func GetDense(r, c int) *mat.Dense {
	return getMatrixPool(d{r: r, c: c}).Get().(*mat.Dense)
}

func PutDense(m *mat.Dense) {
	r, c := m.Dims()
	getMatrixPool(d{r: r, c: c}).Put(m)
}

func GetVec(n int) *mat.VecDense {
	return getVecPool(n).Get().(*mat.VecDense)
}

func PutVec(v *mat.VecDense) {
	getVecPool(v.Len()).Put(v)
}
