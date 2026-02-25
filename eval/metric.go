package eval

// Metric defines the interface for evaluating model performance.
// Any type that implements this interface can be used to compare true values against predicted values.
type Metric interface {
	Name() string
	Compute(yTrue, yPredict []float64) (float64, error)
}

var (
	MeanAbsoluteError    = MAE{}
	MeanSquaredError     = MSE{}
	RootMeanSquaredError = RMSE{}
	RSquared             = R2{}
)

var (
	// RegressionMetrics is the default collection of regression evaluation metrics.
	//
	// It contains commonly used metrics for assessing regression model performance, including MAE, MSE, RMSE, and R².
	//
	// This slice can be used to iterate over multiple metrics when evaluating predictions against true values.
	RegressionMetrics = []Metric{
		MeanAbsoluteError,
		MeanSquaredError,
		RootMeanSquaredError,
		RSquared,
	}
)

// NameRegressionMetrics returns the names of all metrics defined in the RegressionMetrics slice.
func NameRegressionMetrics() []string {
	names := make([]string, 0, len(RegressionMetrics))

	for _, m := range RegressionMetrics {
		names = append(names, m.Name())
	}

	return names
}
