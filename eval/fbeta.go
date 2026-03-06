package eval

import (
	"errors"
	"fmt"
)

// FBeta implements the Metric interface for the generalized F-beta score.
//
// It allows weighting precision and recall differently based on the beta parameter.
type FBeta struct {
	beta float64
}

// NewFBeta creates a new FBeta metric.
//
// beta > 1 favors Recall, beta < 1 favors Precision, and beta = 1 is the standard F1-Score.
func NewFBeta(b float64) *FBeta {
	return &FBeta{
		beta: b,
	}
}

// Name returns the formatted name of the metric based on its beta value.
func (f *FBeta) Name() string {
	if f.beta == 1.0 {
		return "f1-score"
	}

	return fmt.Sprintf("f%.1f-score", f.beta)
}

// Compute calculates the Macro-Averaged F-beta score.
func (f *FBeta) Compute(yTrue, yPredict []float64) (float64, error) {
	lenTrue := len(yTrue)
	if lenTrue == 0 {
		return 0, errors.New("empty slice")
	}

	if lenTrue != len(yPredict) {
		return 0, errors.New("length mismatch")
	}

	classes := make(map[float64]bool)
	for _, v := range yTrue {
		classes[v] = true
	}

	var total float64
	bSquared := f.beta * f.beta

	for class := range classes {
		var tp, fp, fn float64

		for i := range yTrue {
			if yTrue[i] == class {
				if yPredict[i] == class {
					tp++
				} else {
					fn++
				}
			} else if yPredict[i] == class {
				fp++
			}
		}

		var (
			precision = 0.0
			recall    = 0.0
		)

		positiveCount := tp + fp
		if positiveCount > 0 {
			precision = tp / positiveCount
		}

		trueCount := tp + fn
		if trueCount > 0 {
			recall = tp / trueCount
		}

		if (precision + recall) > 0 {
			total += (1 + bSquared) * (precision * recall) / ((bSquared * precision) + recall)
		}
	}

	return total / float64(len(classes)), nil
}

// Beta returns the current weight factor beta.
func (f *FBeta) Beta() float64 {
	return f.beta
}

// SetBeta updates the beta value.
//
// It returns an error if the provided beta is negative, as beta must be >= 0 for the F-beta formula to be valid.
func (f *FBeta) SetBeta(b float64) error {
	if b < 0 {
		return fmt.Errorf("beta must be non-negative, got %f", b)
	}

	f.beta = b
	return nil
}
