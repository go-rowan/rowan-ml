package cluster

func CalculateWCSS(centroids, X [][]float64, clusters [][]int, measure func(a, b []float64) (float64, error)) (float64, error) {
	var inertia float64

	for i, cluster := range clusters {
		centroid := centroids[i]

		for _, point := range cluster {
			d, err := measure(X[point], centroid)
			if err != nil {
				return 0, err
			}

			inertia += d * d
		}
	}

	return inertia, nil
}
