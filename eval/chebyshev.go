package eval

import (
	"errors"
	"math"
)

// Chebyshev represents the L∞ norm distance, also known as the chessboard distance.
//
// It calculates the distance between two points as the maximum absolute difference along any single dimension.
type ChebyshevDistance struct{}

// Name returns the canonical name of the distance metric.
func (ChebyshevDistance) Name() string {
	return "chebyshev"
}

// Compute calculates the Chebyshev distance between two float64 slices.
//
// It returns an error if the slices have different lengths to prevent undefined behavior.
func (ChebyshevDistance) Compute(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("length mismatch")
	}

	var max float64
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if d > max {
			max = d
		}
	}

	return max, nil
}
