package eval

import "errors"

// MSE implements the Mean Squared Error metric.
type MSE struct{}

// Name returns the string identifier of the metric.
func (MSE) Name() string { return "mse" }

// Compute calculates the Mean Squared Error for the given data sets.
// The formula used is: (1/n) * Σ(yTrue - yPredict)².
func (MSE) Compute(yTrue, yPredict []float64) (float64, error) {
	lenTrue := len(yTrue)
	if lenTrue == 0 {
		return 0, errors.New("empty slice")
	}

	if lenTrue != len(yPredict) {
		return 0, errors.New("length mismatch")
	}

	sum := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPredict[i]
		sum += diff * diff
	}

	return sum / float64(lenTrue), nil
}
