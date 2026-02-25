package mathx

func LinearGradients(X [][]float64, y, weights []float64, bias float64) ([]float64, float64, error) {
	nSamples := float64(len(X))
	nFeatures := len(weights)

	dW := make([]float64, nFeatures)
	dB := 0.0

	for i := range X {
		p, err := PredictLinear(X[i], weights, bias)
		if err != nil {
			return nil, 0, err
		}

		e := p - y[i]

		dB += e

		for j := 0; j < nFeatures; j++ {
			dW[j] += e * X[i][j]
		}
	}

	for j := 0; j < nFeatures; j++ {
		dW[j] /= nSamples
	}
	dB /= nSamples

	return dW, dB, nil
}
