package classification

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan-ml/internal/slice"
	"github.com/go-rowan/rowan/table"
)

// CategoricalNaiveBayes implements the Naive Bayes algorithm for discrete features.
//
// This model is suitable for features that represent discrete categories or counts. It estimates probabilities based on the frequency of each category within each class.
type CategoricalNaiveBayes struct {
	priors          []float64
	counts          [][]map[any]float64
	classLabels     []int
	options         *naiveBayesOptions
	features        []string
	featureIndexMap map[string]int
	fitted          bool

	classSampleCounts []float64
	nCategories       map[int]map[int]float64 // [classIdx][featureIdx]
}

// NewCategoricalNaiveBayes initializes an empty CategoricalNaiveBayes model.
//
// The model must be populated via the Fit method before it can perform any predictions or inference.
func NewCategoricalNaiveBayes(options ...NaiveBayesOption) *CategoricalNaiveBayes {
	opts := &naiveBayesOptions{
		alpha: 1.0,
	}

	for _, o := range options {
		o(opts)
	}

	return &CategoricalNaiveBayes{
		options:  opts,
		features: []string{},
		fitted:   false,
	}
}

// Fit trains the Categorical Naive Bayes model by analyzing the frequency distribution of categorical features across different classes.
//
// The training process performs the following:
//  1. Statistics Gathering: Calculates class priors and sample counts per class.
//  2. Categorical Mapping: Uses a multi-dimensional map [class][feature][value] to count occurrences of discrete category values.
//  3. Pre-computation: Calculates and stores the number of unique categories (nCategories) per feature/class to optimize the Laplace smoothing calculation during inference.
//
// Note: All input columns in table 'x' must be categorical.
func (cnb *CategoricalNaiveBayes) Fit(x, y *rowan.Table) error {
	if x == nil || y == nil {
		return errors.New("x and y must not be nil")
	}

	sampleCount := x.Len()
	if sampleCount != y.Len() {
		return fmt.Errorf("features and target tables must have the same number of rows (%d vs %d)", sampleCount, y.Len())
	}

	yData, err := y.ColumnToIntSlice(y.Columns()[0])
	if err != nil {
		return err
	}

	classMap := slice.MapIntSliceIndices(yData)
	classCount := len(classMap)

	cnb.classLabels = make([]int, 0, classCount)
	cnb.classSampleCounts = make([]float64, classCount)
	labelToIdx := make(map[int]int)
	cnb.priors = make([]float64, classCount)
	cnb.counts = make([][]map[any]float64, classCount)
	cnb.nCategories = make(map[int]map[int]float64)

	columns := x.Columns()
	featureCount := len(columns)

	for label, indices := range classMap {
		classIdx := len(cnb.classLabels)

		cnb.classLabels = append(cnb.classLabels, label)
		labelToIdx[label] = classIdx

		cnb.classSampleCounts[classIdx] = float64(len(indices))

		cnb.priors[classIdx] = float64(len(indices)) / float64(sampleCount)

		cnb.counts[classIdx] = make([]map[any]float64, featureCount)
		for i := range cnb.counts[classIdx] {
			cnb.counts[classIdx][i] = make(map[any]float64)
		}

		cnb.nCategories[classIdx] = make(map[int]float64)
	}

	for i := range featureCount {
		col, err := x.Col(columns[i])
		if err != nil {
			return err
		}

		if !col.Categorical() {
			return fmt.Errorf("column %s is not categorical", columns[i])
		}

		colData := col.Values()

		for label, indices := range classMap {
			classIdx := labelToIdx[label]

			for _, rowIdx := range indices {
				categoryValue := colData[rowIdx]

				cnb.counts[classIdx][i][categoryValue]++
			}

			nCtg := float64(len(cnb.counts[classIdx][i]))
			if nCtg == 0 {
				nCtg = 1
			}
			cnb.nCategories[classIdx][i] = nCtg
		}
	}

	cnb.features = columns

	cnb.featureIndexMap = make(map[string]int)
	for i, name := range cnb.features {
		cnb.featureIndexMap[name] = i
	}

	cnb.fitted = true

	return nil
}

// Predict performs inference on a new dataset using the learned categorical probability distributions.
//
// Inference is performed in the log-domain for numerical stability.
// For each observation, the model calculates the posterior probability using Additive (Laplace) Smoothing to handle categories with zero frequency or values not seen during training.
// The smoothing formula used: P(xi|C) = (count + alpha) / (n + alpha * nCategories).
//
// It returns a table with the predicted class labels in the "y_pred" column.
func (cnb *CategoricalNaiveBayes) Predict(x *rowan.Table) (*rowan.Table, error) {
	if !cnb.fitted {
		return nil, errors.New("model is not fitted")
	}

	featuresData := make([][]any, len(cnb.features))
	for fIdx, feature := range cnb.features {
		col, err := x.Col(feature)
		if err != nil {
			return nil, err
		}

		featuresData[fIdx] = col.Values()
	}

	classCount := len(cnb.classLabels)
	sampleCount := x.Len()
	yPred := make([]any, sampleCount)

	for i := range sampleCount {
		bestIdx := -1
		maxLogProb := -math.MaxFloat64

		for cIdx := range classCount {
			logProb := math.Log(cnb.priors[cIdx])

			for fIdx := range cnb.features {
				value := featuresData[fIdx][i]
				count := cnb.counts[cIdx][fIdx][value]

				nCategories := cnb.nCategories[cIdx][fIdx]

				n := cnb.classSampleCounts[cIdx]

				alpha := cnb.options.alpha
				prob := (count + alpha) / (n + alpha*float64(nCategories))
				logProb += math.Log(prob)
			}

			if logProb > maxLogProb {
				maxLogProb = logProb
				bestIdx = cIdx
			}
		}

		yPred[i] = cnb.classLabels[bestIdx]
	}

	return table.New(map[string][]any{"y_pred": yPred})
}
