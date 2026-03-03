package linear

import "github.com/go-rowan/rowan-ml/preprocess"

type linearRegressionOptions struct {
	learnRate float64
	epochs    int
	scaler    preprocess.Transformer
}

// LinearRegressionOption defines a function type for configuring LinearRegression parameters.
type LinearRegressionOption func(*linearRegressionOptions)

// WithLearnRate sets the learning rate for the gradient descent optimizer.
func WithLearnRate(rate float64) LinearRegressionOption {
	return func(o *linearRegressionOptions) { o.learnRate = rate }
}

// WithEpochs sets the number of iterations for the training process.
func WithEpochs(epochs int) LinearRegressionOption {
	return func(o *linearRegressionOptions) {
		o.epochs = epochs
	}
}

// WithScaler sets a transformer to scale features before fitting and prediction.
func WithScaler(scaler preprocess.Transformer) LinearRegressionOption {
	return func(o *linearRegressionOptions) {
		o.scaler = scaler
	}
}
