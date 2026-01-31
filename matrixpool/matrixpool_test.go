package matrixpool

import (
	"sync"
	"testing"
)

func TestGetPutDense(t *testing.T) {
	// Get a matrix from the pool
	m := GetDense(3, 4)
	if m == nil {
		t.Fatal("GetDense returned nil")
	}

	r, c := m.Dims()
	if r != 3 || c != 4 {
		t.Errorf("Expected dimensions (3, 4), got (%d, %d)", r, c)
	}

	// Verify matrix is zeroed
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if m.At(i, j) != 0 {
				t.Errorf("Expected zero at (%d, %d), got %f", i, j, m.At(i, j))
			}
		}
	}

	// Modify the matrix
	m.Set(0, 0, 42.0)

	// Return to pool
	PutDense(m)

	// Get again - should be same object but reset
	m2 := GetDense(3, 4)
	if m2 != m {
		t.Error("Expected to get same matrix from pool")
	}

	// Verify it's been reset to zero
	if m2.At(0, 0) != 0 {
		t.Errorf("Expected reset matrix, got %f at (0,0)", m2.At(0, 0))
	}

	PutDense(m2)
}

func TestGetPutVec(t *testing.T) {
	// Get a vector from the pool
	v := GetVec(5)
	if v == nil {
		t.Fatal("GetVec returned nil")
	}

	if v.Len() != 5 {
		t.Errorf("Expected length 5, got %d", v.Len())
	}

	// Verify vector is zeroed
	for i := 0; i < v.Len(); i++ {
		if v.AtVec(i) != 0 {
			t.Errorf("Expected zero at index %d, got %f", i, v.AtVec(i))
		}
	}

	// Modify the vector
	v.SetVec(0, 99.0)

	// Return to pool
	PutVec(v)

	// Get again - should be same object but zeroed
	v2 := GetVec(5)
	if v2 != v {
		t.Error("Expected to get same vector from pool")
	}

	// Verify it's been zeroed
	if v2.AtVec(0) != 0 {
		t.Errorf("Expected zeroed vector, got %f at index 0", v2.AtVec(0))
	}

	PutVec(v2)
}

func TestMultipleDimensions(t *testing.T) {
	// Different dimensions should have different pools
	m1 := GetDense(2, 3)
	m2 := GetDense(4, 5)
	m3 := GetDense(2, 3) // Same as m1

	PutDense(m1)
	PutDense(m2)
	PutDense(m3)

	// Same dimensions should reuse
	m4 := GetDense(2, 3)
	if m4 != m1 && m4 != m3 {
		t.Error("Expected to reuse matrix from pool with same dimensions")
	}

	PutDense(m4)
}

func TestConcurrentAccess(t *testing.T) {
	// Test concurrent access to ensure no races
	var wg sync.WaitGroup
	const goroutines = 100
	const iterations = 100

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				// Test Dense
				m := GetDense(3, 3)
				m.Set(0, 0, 1.0)
				PutDense(m)

				// Test Vec
				v := GetVec(5)
				v.SetVec(0, 2.0)
				PutVec(v)
			}
		}()
	}

	wg.Wait()
}

func TestResetBehavior(t *testing.T) {
	// Ensure Reset() properly clears Dense matrix
	m := GetDense(2, 2)
	m.Set(0, 0, 1.0)
	m.Set(1, 1, 2.0)
	m.Reset()

	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if m.At(i, j) != 0 {
				t.Errorf("Reset() didn't zero element (%d, %d): got %f", i, j, m.At(i, j))
			}
		}
	}

	PutDense(m)
}

func TestZeroBehavior(t *testing.T) {
	// Ensure Zero() properly clears VecDense
	v := GetVec(3)
	v.SetVec(0, 1.0)
	v.SetVec(1, 2.0)
	v.SetVec(2, 3.0)
	v.Zero()

	for i := 0; i < v.Len(); i++ {
		if v.AtVec(i) != 0 {
			t.Errorf("Zero() didn't zero element %d: got %f", i, v.AtVec(i))
		}
	}

	PutVec(v)
}
