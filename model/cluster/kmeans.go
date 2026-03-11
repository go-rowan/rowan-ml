package cluster

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/go-rowan/rowan"
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

	for i := 0; i < k.maxIter; i++ {
		clusters := make([][]int, k.k)

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

	k.fitted = true

	return nil
}

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
