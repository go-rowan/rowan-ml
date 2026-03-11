package model

import "github.com/go-rowan/rowan"

// SupervisedLearner defines the interface for models that require labeled training data.
// Implementations are expected to map input features (x) to corresponding targets (y) via the Fit method.
type SupervisedLearner interface {
	Fit(x, y *rowan.Table) error
}

// UnsupervisedLearner defines the interface for models that learn patterns directly from input data without explicit target labels.
type UnsupervisedLearner interface {
	Fit(x *rowan.Table) error
}

// Predictor defines the interface for models that can make predictions.
// It takes a table of features (x) and returns a table of predicted values.
type Predictor interface {
	Predict(x *rowan.Table) (*rowan.Table, error)
}
