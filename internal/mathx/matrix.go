package mathx

func CopyMatrix(m [][]float64) [][]float64 {
	mtrx := make([][]float64, len(m))

	for i := range m {
		mtrx[i] = make([]float64, len(m[i]))
		copy(mtrx[i], m[i])
	}

	return mtrx
}
