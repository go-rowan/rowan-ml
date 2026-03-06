package eval

import "errors"

// Acc implements the Metric interface for calculating classification accuracy.
//
// Accuracy is the simplest performance measure, representing the ratio of correctly predicted observations to the total observations.
type Acc struct{}

// Name returns the string identifier of the metric.
func (Acc) Name() string {
	return "accuracy"
}

// Compute calculates the accuracy score between true labels and predicted labels.
// It returns a value between 0.0 and 1.0, where 1.0 indicates perfect classification.
//
// An error is returned if the input slices are empty or have mismatched lengths.
func (Acc) Compute(yTrue, yPredict []float64) (float64, error) {
	lenTrue := len(yTrue)
	if lenTrue == 0 {
		return 0, errors.New("empty slice")
	}

	if lenTrue != len(yPredict) {
		return 0, errors.New("length mismatch")
	}

	var correct int
	for i := range yTrue {
		if yTrue[i] == yPredict[i] {
			correct++
		}
	}

	return float64(correct) / float64(lenTrue), nil
}
