package clustering

import "math"

func (k *KMeans) findClosestCentroid(row []float64) (int, error) {
	closest := -1
	min := math.MaxFloat64

	for i, centroid := range k.centroids {
		d, err := k.options.distance.Measure(row, centroid)
		if err != nil {
			return -1, err
		}

		if d < min {
			min = d
			closest = i
		}
	}

	return closest, nil
}

func (k *KMeans) calculateNewCentroids(X [][]float64, clusters [][]int) [][]float64 {
	newCentroids := make([][]float64, k.k)
	count := len(X[0])

	for i := 0; i < k.k; i++ {
		newCentroids[i] = make([]float64, count)

		if len(clusters[i]) == 0 {
			newCentroids[i] = k.centroids[i]
			continue
		}

		for _, idx := range clusters[i] {
			for j := 0; j < count; j++ {
				newCentroids[i][j] += X[idx][j]
			}
		}

		for j := 0; j < count; j++ {
			newCentroids[i][j] /= float64(len(clusters[i]))
		}
	}

	return newCentroids
}

func (k *KMeans) isConverged(old, new [][]float64) (bool, error) {
	for i := 0; i < k.k; i++ {
		d, err := k.options.distance.Measure(old[i], new[i])
		if err != nil {
			return false, err
		}

		if d > k.tolerance {
			return false, nil
		}
	}

	return true, nil
}
