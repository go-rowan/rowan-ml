package eval

import (
	"errors"
	"fmt"
	"math"
)

// Minkowski represents the Lp norm distance in a normed vector space.
//
// It provides a generalized formula for calculating the distance between two points by parameterizing the exponent p.
type MinkowskiDistance struct {
	p float64
}

// NewMinkowskiDistance creates a new MinkowskiDistance with a specific p-order.
//
// The parameter p must be at least 1 to satisfy the Minkowski inequality and maintain the properties of a metric space.
func NewMinkowskiDistance(p float64) (*MinkowskiDistance, error) {
	if p < 1 {
		return nil, fmt.Errorf("p must be greater than or equal to 1, got %f", p)
	}

	return &MinkowskiDistance{p: p}, nil
}

// Name returns the canonical name of the distance metric including its p-value.
func (md *MinkowskiDistance) Name() string {
	return fmt.Sprintf("minkowski(p=%.2f)", md.p)
}

// Measure calculates the Minkowski distance between two vectors.
func (md *MinkowskiDistance) Measure(a, b []float64) (float64, error) {
	switch md.p {
	case 2:
		return Euclidean.Measure(a, b)
	case 1:
		return Manhattan.Measure(a, b)
	}

	if len(a) != len(b) {
		return 0, errors.New("length mismatch")
	}

	var sum float64
	for i := range a {
		diff := math.Abs(a[i] - b[i])
		sum += math.Pow(diff, md.p)
	}

	return math.Pow(sum, 1/md.p), nil
}

// P returns the current exponent value p.
func (md *MinkowskiDistance) P() float64 {
	return md.p
}

// SetP updates the exponent value p.
// It returns an error if the provided p < 1.
func (md *MinkowskiDistance) SetP(p float64) error {
	if p < 1 {
		return fmt.Errorf("p must be greater than or equal to 1, got %f", p)
	}

	md.p = p
	return nil
}
