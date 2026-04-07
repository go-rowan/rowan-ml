package linear

import (
	"errors"
	"fmt"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan-ml/internal/mathx"
	"github.com/go-rowan/rowan/table"
)

// LinearRegression implements a linear model with coefficients to minimize the residual sum of squares between the observed targets and the predictions.
type LinearRegression struct {
	weights  []float64
	bias     float64
	features []string
	fitted   bool
	options  *linearRegressionOptions
}

// NewRegression creates and returns a new LinearRegression model with default or custom options.
// Default values: learnRate=0.01, epochs=1000.
func NewRegression(options ...LinearRegressionOption) *LinearRegression {
	opts := &linearRegressionOptions{
		learnRate: 0.01,
		epochs:    1000,
	}

	for _, o := range options {
		o(opts)
	}

	return &LinearRegression{
		options:  opts,
		features: []string{},
		fitted:   false,
	}
}

// Fit trains the linear model using gradient descent.
// It performs feature scaling if a scaler was provided via WithScaler.
func (lr *LinearRegression) Fit(x, y *rowan.Table) error {
	if x == nil || y == nil {
		return errors.New("x and y must not be nil")
	}

	if x.Len() != y.Len() {
		return fmt.Errorf("features and target tables must have the same number of rows (%d vs %d)", x.Len(), y.Len())
	}

	if lr.options.scaler != nil {
		err := lr.options.scaler.Fit(x, x.Columns()...)
		if err != nil {
			return fmt.Errorf("scaling error during fit: %w", err)
		}

		x, err = lr.options.scaler.Transform(x)
		if err != nil {
			return fmt.Errorf("scaling error during transform: %w", err)
		}
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return err
	}

	yCol, err := y.NumericSlice(0)
	if err != nil {
		return err
	}

	featuresCount := len(X[0])

	lr.weights = make([]float64, featuresCount)
	lr.bias = 0

	for e := 0; e < lr.options.epochs; e++ {
		dW, dB, err := mathx.LinearGradients(X, yCol, lr.weights, lr.bias)
		if err != nil {
			return err
		}

		lr.weights, lr.bias = mathx.SGDStep(lr.weights, lr.bias, dW, dB, lr.options.learnRate)
	}

	lr.features = x.Columns()
	lr.fitted = true

	return nil
}

// Predict generates predictions for the input table.
// The model must be fitted before calling this method.
func (lr *LinearRegression) Predict(x *rowan.Table) (*rowan.Table, error) {
	if x == nil {
		return nil, errors.New("x is nil")
	}

	if !lr.fitted {
		return nil, errors.New("model is not fitted yet")
	}

	inputCols := x.Columns()
	if len(inputCols) != len(lr.features) {
		return nil, errors.New("feature count mismatch")
	}
	for i := range inputCols {
		if inputCols[i] != lr.features[i] {
			return nil, fmt.Errorf("feature mismatch in index %d: expected %s, got %s", i, lr.features[i], inputCols[i])
		}
	}

	var err error
	if lr.options.scaler != nil {
		x, err = lr.options.scaler.Transform(x)
		if err != nil {
			return nil, fmt.Errorf("scaling error during transform: %w", err)
		}
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return nil, err
	}

	sampleCount := len(X)
	if sampleCount == 0 {
		return nil, errors.New("x has no rows")
	}

	if len(X[0]) != len(lr.weights) {
		return nil, errors.New("features count mismatch")
	}

	yPred := make([]any, sampleCount)

	for i := 0; i < sampleCount; i++ {
		d, err := mathx.Dot(X[i], lr.weights)
		if err != nil {
			return nil, err
		}

		yPred[i] = d + lr.bias
	}

	return table.New(map[string][]any{"y_pred": yPred})
}

// IsFitted returns true if the model has been successfully trained.
func (lr *LinearRegression) IsFitted() bool {
	return lr.fitted
}

// Features returns the names of the features the model was trained on.
func (lr *LinearRegression) Features() []string {
	features := make([]string, len(lr.features))
	copy(features, lr.features)

	return features
}

// WeightsMap returns a map of feature names to their respective weights.
func (lr *LinearRegression) WeightsMap() map[string]float64 {
	if !lr.fitted {
		return nil
	}

	weightsMap := make(map[string]float64, len(lr.features))
	for i, name := range lr.features {
		weightsMap[name] = lr.weights[i]
	}

	return weightsMap
}

// Weights returns a copy of the weights learned by the model.
func (lr *LinearRegression) Weights() []float64 {
	lenWeights := len(lr.weights)
	weights := make([]float64, lenWeights)

	if lenWeights > 0 {
		copy(weights, lr.weights)
	}

	return weights
}

// Weight returns the weight of a specific feature by its name.
// It returns an error if the feature name is not found or the model is not fitted.
func (lr *LinearRegression) Weight(feature string) (float64, error) {
	if !lr.fitted {
		return 0, errors.New("model is not fitted")
	}

	for i, name := range lr.features {
		if name == feature {
			return lr.weights[i], nil
		}
	}

	return 0, fmt.Errorf("feature %s not found in model", feature)
}

// Bias returns the bias term learned by the model.
func (lr *LinearRegression) Bias() float64 {
	return lr.bias
}
