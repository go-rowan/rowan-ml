package eval

import (
	"errors"
)

// R2 implements the Coefficient of Determination (R-squared) metric.
type R2 struct{}

// Name returns the string identifier of the metric.
func (R2) Name() string { return "r2" }

// Compute calculates the R2 score for the given data sets.
// It measures the proportion of variance in the dependent variable that is predictable from the independent variable(s).
func (R2) Compute(yTrue, yPredict []float64) (float64, error) {
	lenTrue := len(yTrue)
	if lenTrue == 0 {
		return 0, errors.New("empty slice")
	}

	if lenTrue != len(yPredict) {
		return 0, errors.New("length mismatch")
	}

	var (
		sumTrue = 0.0
		rss     = 0.0
		tss     = 0.0
	)

	for _, y := range yTrue {
		sumTrue += y
	}
	meanTrue := sumTrue / float64(lenTrue)

	for i := range yTrue {
		e := yTrue[i] - yPredict[i]
		rss += e * e

		t := yTrue[i] - meanTrue
		tss += t * t
	}

	if tss == 0 {
		return 0, nil
	}

	return 1 - (rss / tss), nil
}
