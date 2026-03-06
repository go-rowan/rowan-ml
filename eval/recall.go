package eval

import "errors"

// Rec implements the Metric interface for Recall calculation.
//
// Recall (Sensitivity) is the ratio of correctly predicted positive observations to all actual observations in that class (TP / (TP + FN)).
type Rec struct{}

// Name returns the string identifier of the metric.
func (Rec) Name() string {
	return "recall"
}

// Compute calculates the Macro-Averaged Recall for the given predictions.
func (Rec) Compute(yTrue, yPredict []float64) (float64, error) {
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

	for class := range classes {
		var tp, fn float64

		for i := range yTrue {
			if yTrue[i] == class {
				if yPredict[i] == class {
					tp++
				} else {
					fn++
				}
			}
		}

		count := tp + fn
		if count > 0 {
			total += tp / count
		}
	}

	return total / float64(len(classes)), nil
}
