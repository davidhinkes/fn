package matrixpool

import (
	"sync"
	"testing"
)

func TestGetPutDense(t *testing.T) {
	m := GetDense(3, 4)
	if m == nil {
		t.Fatal("GetDense returned nil")
	}

	r, c := m.Dims()
	if r != 3 || c != 4 {
		t.Errorf("Expected dimensions (3, 4), got (%d, %d)", r, c)
	}

	m.Set(0, 0, 42.0)
	PutDense(m)
}

func TestGetPutVec(t *testing.T) {
	v := GetVec(5)
	if v == nil {
		t.Fatal("GetVec returned nil")
	}

	if v.Len() != 5 {
		t.Errorf("Expected length 5, got %d", v.Len())
	}

	v.SetVec(0, 99.0)
	PutVec(v)
}

func TestConcurrentAccess(t *testing.T) {
	var wg sync.WaitGroup
	const goroutines = 100
	const iterations = 100

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				m := GetDense(3, 3)
				m.Set(0, 0, 1.0)
				PutDense(m)

				v := GetVec(5)
				v.SetVec(0, 2.0)
				PutVec(v)
			}
		}()
	}

	wg.Wait()
}
