package neighbors

import (
	"errors"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan-ml/internal/neighbor"
	"github.com/go-rowan/rowan/table"
)

// KNNRegressor represents a K-Nearest Neighbors model for regression tasks. It stores training data in-memory and performs lazy learning.
//
// The model performance depends on the chosen distance metric and feature scaling.
type KNNRegressor struct {
	k        int
	trainX   [][]float64
	trainY   []float64
	features []string
	fitted   bool
	options  *knnOptions
}

// NewKNNRegressor initializes a new regression model with the specified K value and optional configurations.
//
// It applies a default set of options which can be overridden using KNNOption functions.
func NewKNNRegressor(k int, options ...KNNOption) *KNNRegressor {
	opts := defaultOptions()
	for _, o := range options {
		o(opts)
	}

	return &KNNRegressor{
		k:       k,
		options: opts,
	}
}

// Fit trains the KNNRegressor by storing the training data and preparing the scaling parameters if a scaler is provided in the options.
//
// It validates that both feature and target tables have consistent row counts and converts the raw table data into a numerical format suitable for distance calculations.
// This process is a "lazy learning" step where data is stored rather than used for optimization.
func (kr *KNNRegressor) Fit(x, y *rowan.Table) error {
	X, Y, err := fit(x, y, kr.options.scaler)
	if err != nil {
		return err
	}

	kr.features = x.Columns()
	kr.trainX = X
	kr.trainY = Y

	kr.fitted = true

	return nil
}

// Predict estimates continuous target values for the given input table by finding the K-nearest neighbors for each row.
//
// For every instance in the input, the method calculates the distances to all training samples and identifies the top K closest neighbors. The final prediction is the arithmetic mean of the target values of these neighbors.
//
// If a scaler was used during fitting, the input table will be transformed accordingly before making predictions.
func (kr *KNNRegressor) Predict(x *rowan.Table) (*rowan.Table, error) {
	if x == nil {
		return nil, errors.New("x is nil")
	}

	if !kr.fitted {
		return nil, errors.New("model is not fitted yet")
	}

	if kr.options.scaler != nil {
		var err error

		x, err = kr.options.scaler.Transform(x)
		if err != nil {
			return nil, err
		}
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return nil, err
	}

	yPred := make([]any, len(X))

	for i, row := range X {
		indices, err := neighbor.GetKNearest(kr.k, row, kr.trainX, kr.options.distance)
		if err != nil {
			return nil, err
		}

		var sum float64

		for _, idx := range indices {
			sum += kr.trainY[idx]
		}

		yPred[i] = sum / float64(len(indices))
	}

	return table.New(map[string][]any{"y_pred": yPred})
}

// IsFitted returns true if the model has been successfully trained.
func (kr *KNNRegressor) IsFitted() bool {
	return kr.fitted
}

// Features returns the names of the features the model was trained on.
func (kr *KNNRegressor) Features() []string {
	features := make([]string, len(kr.features))
	copy(features, kr.features)

	return features
}
