package cluster

import "math/rand"

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
