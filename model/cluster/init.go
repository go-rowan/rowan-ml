package cluster

import "math/rand"

func (k *KMeans) initRandom(X [][]float64) [][]float64 {
	centroids := make([][]float64, k.k)

	p := rand.Perm(len(X))

	for i := 0; i < k.k; i++ {
		centroids[i] = make([]float64, len(X[0]))
		copy(centroids[i], X[p[i]])
	}

	return centroids
}
