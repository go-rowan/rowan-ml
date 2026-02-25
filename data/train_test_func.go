package data

import (
	"errors"
	"math/rand"
	"time"

	"github.com/go-rowan/rowan"
)

type trainTestOptions struct {
	shuffle bool
}

type TrainTestOption func(*trainTestOptions)

// WithNoShuffle disables the random shuffling of rows before splitting.
// By default, rows are shuffled to ensure a representative distribution of data.
func WithNoShuffle() TrainTestOption {
	return func(o *trainTestOptions) {
		o.shuffle = false
	}
}

// TrainTest splits a single table into training and testing sets based on the provided trainSize.
// The trainSize parameter must be between 0.0 and 1.0. If shuffling is enabled (default), it uses the current time as a seed for randomness.
func TrainTest(t *rowan.Table, trainSize float64, options ...TrainTestOption) (*rowan.Table, *rowan.Table, error) {
	if t == nil {
		return nil, nil, errors.New("table is nil")
	}

	if trainSize < 0.0 || trainSize > 1.0 {
		return nil, nil, errors.New("trainSize must be between 0 and 1")
	}

	opts := &trainTestOptions{
		shuffle: true,
	}
	for _, o := range options {
		o(opts)
	}

	indices := getTrainTestIndices(t.Len(), opts.shuffle)

	return splitByIndices(t, indices, trainSize)
}

// TrainTestXY splits a pair of tables (features X and targets Y) into training and testing sets.
//
// The returned tables will be in this order: xTrain, yTrain, xTest, yTest
func TrainTestXY(x, y *rowan.Table, trainSize float64, options ...TrainTestOption) (
	*rowan.Table, *rowan.Table, *rowan.Table, *rowan.Table, error,
) {
	if x == nil || y == nil {
		return nil, nil, nil, nil, errors.New("table x or y is nil")
	}

	lenX := x.Len()
	if lenX != y.Len() {
		return nil, nil, nil, nil, errors.New("x and y must have the same number of rows")
	}

	opts := &trainTestOptions{
		shuffle: true,
	}
	for _, o := range options {
		o(opts)
	}

	indices := getTrainTestIndices(lenX, opts.shuffle)

	xTrain, xTest, err := splitByIndices(x, indices, trainSize)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	yTrain, yTest, err := splitByIndices(y, indices, trainSize)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return xTrain, yTrain, xTest, yTest, nil
}

func getTrainTestIndices(n int, shuffle bool) []int {
	indices := make([]int, n)

	for i := 0; i < n; i++ {
		indices[i] = i
	}

	if shuffle {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))

		r.Shuffle(n, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	return indices
}

func splitByIndices(t *rowan.Table, indices []int, trainSize float64) (*rowan.Table, *rowan.Table, error) {
	if trainSize < 0.0 || trainSize > 1.0 {
		return nil, nil, errors.New("trainSize must be between 0 and 1")
	}

	n := len(indices)
	nTrain := int(trainSize * float64(n))

	trainTbl, err := t.SelectRows(indices[:nTrain])
	if err != nil {
		return nil, nil, err
	}

	testTbl, err := t.SelectRows(indices[nTrain:])
	if err != nil {
		return nil, nil, err
	}

	return trainTbl, testTbl, nil
}
