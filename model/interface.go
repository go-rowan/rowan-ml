package model

import "github.com/go-rowan/rowan"

// Learner defines the interface for models that can be trained.
// Any type implementing this interface must provide a Fit method to learn from the provided features (x) and targets (y).
type Learner interface {
	Fit(x, y *rowan.Table) error
}

// Predictor defines the interface for models that can make predictions.
// It takes a table of features (x) and returns a table of predicted values.
type Predictor interface {
	Predict(x *rowan.Table) (*rowan.Table, error)
}
