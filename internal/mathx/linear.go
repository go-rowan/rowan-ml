package mathx

import "errors"

func Dot(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("dot: length mismatch")
	}

	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum, nil
}

func PredictLinear(x, weights []float64, bias float64) (float64, error) {
	d, err := Dot(weights, x)
	if err != nil {
		return 0, err
	}

	return d + bias, nil
}
