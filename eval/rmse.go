package eval

import "math"

// RMSE implements the Root Mean Squared Error metric.
type RMSE struct{}

// Name returns the string identifier of the metric.
func (RMSE) Name() string { return "rmse" }

// Compute calculates the Root Mean Squared Error for the given data sets.
// It is the square root of the MSE.
func (RMSE) Compute(yTrue, yPredict []float64) (float64, error) {
	mse := MSE{}

	mseValue, err := mse.Compute(yTrue, yPredict)
	if err != nil {
		return 0, err
	}

	return math.Sqrt(mseValue), nil
}
