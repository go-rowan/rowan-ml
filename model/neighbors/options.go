package neighbors

import (
	"github.com/go-rowan/rowan-ml/eval"
	"github.com/go-rowan/rowan-ml/preprocess"
)

type knnOptions struct {
	distance eval.Distance
	scaler   preprocess.Transformer
}

func defaultOptions() *knnOptions {
	return &knnOptions{
		distance: eval.Euclidean,
	}
}

// KNNOption defines a function type for configuring a KNN model.
type KNNOption func(*knnOptions)

// WithDistance sets the distance metric for the KNN algorithm.
//
// If not provided, the model typically defaults to Euclidean distance.
func WithDistance(d eval.Distance) KNNOption {
	return func(o *knnOptions) {
		o.distance = d
	}
}

// WithScaler attaches a preprocessing transformer (e.g., Z-Score or Min-Max scaler) to the KNN model to normalize features before distance calculation.
func WithScaler(s preprocess.Transformer) KNNOption {
	return func(o *knnOptions) {
		o.scaler = s
	}
}
