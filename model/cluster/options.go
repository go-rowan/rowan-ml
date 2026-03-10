package cluster

import (
	"github.com/go-rowan/rowan-ml/eval"
	"github.com/go-rowan/rowan-ml/preprocess"
)

type kMeansOptions struct {
	// strategy  KMeansInitStrategy
	distance  eval.Distance
	scaler    preprocess.Transformer
	maxIter   int
	seed      int64
	tolerance float64
}

// KMeansOption defines a functional option type for configuring kMeansOptions.
type KMeansOption func(*kMeansOptions)

// type KMeansInitStrategy int

// const (
// 	Random KMeansInitStrategy = iota
// 	KMeansPlusPlus
// )

func defaultOptions() *kMeansOptions {
	return &kMeansOptions{
		distance: eval.Euclidean,
		// strategy: Random,
		maxIter:   300,
		seed:      42,
		tolerance: 1e-4,
	}
}

// WithDistance sets the distance metric used during the centroid assignment phase.
func WithDistance(d eval.Distance) KMeansOption {
	return func(o *kMeansOptions) {
		o.distance = d
	}
}

// WithScaler provides a transformer to normalize or standardize features, ensuring that all dimensions contribute proportionately to the distance calculation.
func WithScaler(s preprocess.Transformer) KMeansOption {
	return func(o *kMeansOptions) {
		o.scaler = s
	}
}

// WithSeed configures the seed for initial centroid selection.
func WithSeed(s int64) KMeansOption {
	return func(o *kMeansOptions) {
		o.seed = s
	}
}
