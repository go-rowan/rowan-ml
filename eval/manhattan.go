package eval

import (
	"errors"
	"math"
)

// Manhattan represents the L1 norm distance, also known as taxicab geometry.
//
// It calculates the distance between two points by summing the absolute differences of their coordinates.
type ManhattanDistance struct{}

// Name returns the canonical name of the distance metric.
func (ManhattanDistance) Name() string {
	return "manhattan"
}

// Measure calculates the Manhattan distance between two float64 slices.
//
// It returns an error if the slices have different lengths to prevent undefined behavior.
func (ManhattanDistance) Measure(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("length mismatch")
	}

	var sum float64
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}

	return sum, nil
}
