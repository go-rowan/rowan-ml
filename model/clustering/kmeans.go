package clustering

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan-ml/internal/cluster"
	"github.com/go-rowan/rowan/table"
)

// KMeans implements the K-Means clustering algorithm.
type KMeans struct {
	k         int
	maxIter   int
	randGen   *rand.Rand
	tolerance float64
	centroids [][]float64
	features  []string
	fitted    bool
	inertia   float64
	options   *kMeansOptions
}

// NewKMeans initializes a KMeans instance with the specified number of clusters and optional functional configurations.
func NewKMeans(k int, options ...KMeansOption) *KMeans {
	opts := defaultOptions()
	for _, o := range options {
		o(opts)
	}

	return &KMeans{
		k:         k,
		maxIter:   opts.maxIter,
		tolerance: opts.tolerance,
		randGen:   rand.New(rand.NewSource(opts.seed)),
		options:   opts,
	}
}

// Fit executes the K-Means clustering algorithm on the provided table.
//
// The process follows these sequential phases:
//  1. Preprocessing: If a scaler is configured, it fits to and transforms the input data.
//  2. Initialization: Initial centroids are selected from the observation space.
//  3. Iterative Optimization:
//     - Assignment: Each observation is mapped to the nearest centroid.
//     - Update: Centroids are recalculated based on the mean of assigned points.
//     - Convergence Check: The process terminates if the movement is within 'tolerance' or 'maxIter' is reached.
//  4. Metric Calculation: Computes the final inertia (WCSS) and marks the model as fitted.
func (k *KMeans) Fit(x *rowan.Table) error {
	if x == nil {
		return errors.New("x must not be nil")
	}

	if k.options.scaler != nil {
		err := k.options.scaler.Fit(x, x.Columns()...)
		if err != nil {
			return fmt.Errorf("scaling error during fit: %w", err)
		}

		x, err = k.options.scaler.Transform(x)
		if err != nil {
			return fmt.Errorf("scaling error during transform: %w", err)
		}
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return err
	}

	k.centroids = k.initRandom(X)

	clusters := make([][]int, k.k)
	for i := 0; i < k.maxIter; i++ {
		for idx, row := range X {
			c, err := k.findClosestCentroid(row)
			if err != nil {
				return err
			}

			clusters[c] = append(clusters[c], idx)
		}

		newCentroids := k.calculateNewCentroids(X, clusters)

		converged, err := k.isConverged(k.centroids, newCentroids)
		if err != nil {
			return err
		}

		if converged {
			break
		}
		k.centroids = newCentroids
	}

	inertia, err := cluster.CalculateWCSS(k.centroids, X, clusters, k.options.distance.Measure)
	if err != nil {
		return err
	}

	k.inertia = inertia
	k.fitted = true

	return nil
}

// Predict performs inference on the provided input table, assigning each observation to the nearest cluster centroid.
//
// If a scaler was provided during initialization, it is automatically applied to the input features to maintain consistency with the training distribution.
// It returns a single-column table containing the assigned cluster indices.
func (k *KMeans) Predict(x *rowan.Table) (*rowan.Table, error) {
	if x == nil {
		return nil, errors.New("x must not be nil")
	}

	if !k.fitted {
		return nil, errors.New("model is not fitted yet")
	}

	if k.options.scaler != nil {
		var err error

		x, err = k.options.scaler.Transform(x)
		if err != nil {
			return nil, err
		}
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return nil, err
	}

	clusters := make([]any, len(X))
	for i, row := range X {
		c, err := k.findClosestCentroid(row)
		if err != nil {
			return nil, err
		}

		clusters[i] = c
	}

	return table.New(map[string][]any{"cluster": clusters})
}

// K returns the number of clusters currently configured.
func (k *KMeans) K() int {
	return k.k
}

// SetK updates the number of clusters (k) for the model.
//
// This operation is only permitted before the model is fitted. If called on a model where fitted is true, it returns an error.
func (k *KMeans) SetK(newK int) error {
	if k.fitted {
		return errors.New("cannot change k: model is already fitted")
	}

	if newK < 1 {
		return errors.New("k must be at least 1")
	}

	k.k = newK

	return nil
}

// MaxIter returns the current limit on the number of iterations.
func (k *KMeans) MaxIter() int {
	return k.maxIter
}

// SetMaxIter updates the iteration limit, ensuring it is at least 1.
func (k *KMeans) SetMaxIter(m int) {
	if m < 1 {
		k.maxIter = 1
		return
	}

	k.maxIter = m
}

// Tolerance returns the threshold value used to determine convergence.
func (k *KMeans) Tolerance() float64 {
	return k.tolerance
}

// SetTolerance updates the convergence threshold. A value of 0 forces the algorithm to run until maxIter is reached.
func (k *KMeans) SetTolerance(t float64) {
	if t < 0 {
		k.tolerance = 0
		return
	}

	k.tolerance = t
}

// SetSeed re-initializes the pseudo-random number generator with a new seed, affecting subsequent centroid initialization.
func (k *KMeans) SetSeed(s int64) {
	k.options.seed = s

	k.randGen = rand.New(rand.NewSource(s))
}

// Inertia returns the Within-Cluster Sum of Squares (WCSS), representing the sum of squared distances of samples to their closest cluster center.
//
// This metric serves as an internal measure of clustering coherence; lower values typically indicate a more dense and well-separated clustering.
func (k *KMeans) Inertia() float64 {
	return k.inertia
}

// IsFitted returns true if the model has been successfully trained.
func (k *KMeans) IsFitted() bool {
	return k.fitted
}

// Features returns the names of the features the model was trained on.
func (k *KMeans) Features() []string {
	features := make([]string, len(k.features))
	copy(features, k.features)

	return features
}

// Centroids returns a deep copy of all computed cluster centers.
//
// It returns nil if the model has not been fitted.
func (k *KMeans) Centroids() [][]float64 {
	if !k.fitted {
		return nil
	}

	centroids := make([][]float64, len(k.features))
	for i := range k.centroids {
		centroids[i] = make([]float64, len(k.centroids[i]))
		copy(centroids[i], k.centroids[i])
	}

	return centroids
}

// Centroid returns a deep copy of a specific cluster center by its index.
//
// It returns nil if the model is not fitted or if the index is out of bounds.
func (k *KMeans) Centroid(idx int) []float64 {
	if !k.fitted || idx < 0 || idx >= len(k.centroids) {
		return nil
	}

	centroid := make([]float64, len(k.centroids[idx]))
	copy(centroid, k.centroids[idx])

	return centroid
}
