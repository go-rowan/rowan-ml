package mathx

func SGDStep(weights []float64, bias float64, dW []float64, dB, learnRate float64) ([]float64, float64) {
	for i := range weights {
		weights[i] -= learnRate * dW[i]
	}

	bias -= learnRate * dB

	return weights, bias
}
