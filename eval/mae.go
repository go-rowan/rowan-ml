package eval

import (
	"errors"
	"math"
)

// MAE implements the Mean Absolute Error metric.
type MAE struct{}

// Name returns the string identifier of the metric.
func (MAE) Name() string { return "mae" }

// Compute calculates the Mean Absolute Error for the given data sets.
// It returns an error if the slices are empty or if their lengths do not match.
// The formula used is: (1/n) * Σ|yTrue - yPredict|.
func (MAE) Compute(yTrue, yPredict []float64) (float64, error) {
	lenTrue := len(yTrue)
	if lenTrue == 0 {
		return 0, errors.New("empty slice")
	}

	if lenTrue != len(yPredict) {
		return 0, errors.New("length mismatch")
	}

	sum := 0.0
	for i := range yTrue {
		sum += math.Abs(yTrue[i] - yPredict[i])
	}

	return sum / float64(lenTrue), nil
}
