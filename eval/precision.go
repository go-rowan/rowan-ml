package eval

import "errors"

// Prec implements the Metric interface for Precision calculation.
//
// Precision is the ratio of correctly predicted positive observations to the total predicted positives (TP / (TP + FP)).
type Prec struct{}

// Name returns the string identifier of the metric.
func (Prec) Name() string {
	return "precision"
}

// Compute calculates the Macro-Averaged Precision for the given predictions.
func (Prec) Compute(yTrue, yPredict []float64) (float64, error) {
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
		var tp, fp float64

		for i := range yPredict {
			if yPredict[i] == class {
				if yTrue[i] == class {
					tp++
				} else {
					fp++
				}
			}
		}

		count := tp + fp
		if (count) > 0 {
			total += tp / count
		}
	}

	return total / float64(len(classes)), nil
}
