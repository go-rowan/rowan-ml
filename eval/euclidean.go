package eval

import (
	"errors"
	"math"
)

// Euclidean represents the L2 norm distance, calculating the straight-line distance between two points in Euclidean space.
type EuclideanDistance struct{}

// Name returns the canonical name of the distance metric.
func (EuclideanDistance) Name() string {
	return "euclidean"
}

// Compute calculates the Euclidean distance between two float64 slices.
//
// It returns an error if the slices have different lengths to prevent undefined behavior.
func (EuclideanDistance) Compute(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("length mismatch")
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum), nil
}
