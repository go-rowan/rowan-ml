package neighbors

import (
	"errors"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan-ml/internal/neighbor"
	"github.com/go-rowan/rowan/table"
)

// KNNClassifier represents a K-Nearest Neighbors model for classification tasks. It stores training data in-memory and performs lazy learning.
//
// The model performance depends on the chosen distance metric and feature scaling.
type KNNClassifier struct {
	k        int
	trainX   [][]float64
	trainY   []float64
	features []string
	fitted   bool
	options  *knnOptions
}

// NewKNNClassifier initializes a new KNN model with the specified K value and optional configurations.
//
// It applies a default set of options which can be overridden using KNNOption functions.
func NewKNNClassifier(k int, options ...KNNOption) *KNNClassifier {
	opts := defaultOptions()
	for _, o := range options {
		o(opts)
	}

	return &KNNClassifier{
		k:       k,
		options: opts,
	}
}

// Fit trains the KNN classifier by storing the training data.
//
// It performs data validation and automatically applies feature scaling if a scaler was provided during initialization. As a lazy learner, Fit primarily handles data transformation and storage for later use in Predict.
func (kc *KNNClassifier) Fit(x, y *rowan.Table) error {
	X, Y, err := fit(x, y, kc.options.scaler)
	if err != nil {
		return err
	}

	kc.features = x.Columns()
	kc.trainX = X
	kc.trainY = Y

	kc.fitted = true

	return nil
}

// Predict estimates the class labels for the provided input table.
//
// It transforms the input using the fitted scaler (if any) and identifies the K-nearest neighbors for each row using the configured distance metric.
// In the event of a tie in votes, the label with the nearest average rank among the neighbors is selected as the winner.
func (kc *KNNClassifier) Predict(x *rowan.Table) (*rowan.Table, error) {
	if x == nil {
		return nil, errors.New("x is nil")
	}

	if !kc.fitted {
		return nil, errors.New("model is not fitted yet")
	}

	if kc.options.scaler != nil {
		var err error

		x, err = kc.options.scaler.Transform(x)
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
		indices, err := neighbor.GetKNearest(kc.k, row, kc.trainX, kc.options.distance.Measure)
		if err != nil {
			return nil, err
		}

		votes := make(map[float64]int)
		first := make(map[float64]int)
		max := 0

		for rank, idx := range indices {
			label := kc.trainY[idx]
			votes[label]++

			if _, exists := first[label]; !exists {
				first[label] = rank
			}

			if votes[label] > max {
				max = votes[label]
			}
		}

		var win float64
		earliestRank := len(indices) + 1

		for label, count := range votes {
			if count == max {
				if first[label] < earliestRank {
					earliestRank = first[label]
					win = label
				}
			}
		}

		yPred[i] = win
	}

	return table.New(map[string][]any{"y_pred": yPred})
}

// IsFitted returns true if the model has been successfully trained.
func (kc *KNNClassifier) IsFitted() bool {
	return kc.fitted
}

// Features returns the names of the features the model was trained on.
func (kc *KNNClassifier) Features() []string {
	features := make([]string, len(kc.features))
	copy(features, kc.features)

	return features
}
