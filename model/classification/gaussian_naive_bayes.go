package classification

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan-ml/internal/mathx"
	"github.com/go-rowan/rowan-ml/internal/slice"
	"github.com/go-rowan/rowan/table"
)

// GaussianNaiveBayes implements the Naive Bayes algorithm for continuous features, assuming that the likelihood of the features follows a Gaussian (normal) distribution.
//
// It stores the statistical parameters for each class-feature pair, allowing for efficient probability estimation during inference.
type GaussianNaiveBayes struct {
	means           [][]float64
	variances       [][]float64
	priors          []float64
	classlabels     []int
	features        []string
	featureIndexMap map[string]int
	fitted          bool
}

// NewGaussianNaiveBayes initializes an empty GaussianNaiveBayes model.
//
// The model must be trained using the Fit method before it can be used to perform predictions.
func NewGaussianNaiveBayes() *GaussianNaiveBayes {
	return &GaussianNaiveBayes{
		features: []string{},
		fitted:   false,
	}
}

// Fit trains the Gaussian Naive Bayes model using the provided feature and target tables.
//
// The training process performs Maximum Likelihood Estimation (MLE) to compute:
//  1. Class Priors: The probability of each class based on frequency.
//  2. Feature Means: The average value of each feature per class.
//  3. Feature Variances: The spread of each feature per class, with a small epsilon (1e-9) added to prevent division by zero during inference.
//
// It assumes that the target table contains discrete integer labels representing the classes.
//
// NOTE: The target table (y) must contain discrete values. If float values are provided, they will be truncated to integers (e.g., 1.9 becomes 1).
// Ensure that your class labels remain unique after this conversion to avoid unintended class merging.
func (gnb *GaussianNaiveBayes) Fit(x, y *rowan.Table) error {
	if x == nil || y == nil {
		return errors.New("x and y must not be nil")
	}

	if x.Len() != y.Len() {
		return fmt.Errorf("features and target tables must have the same number of rows (%d vs %d)", x.Len(), y.Len())
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return err
	}

	yData, err := y.ColumnToIntSlice(y.Columns()[0])
	if err != nil {
		return err
	}

	classMap := slice.MapIntSliceIndices(yData)

	classCount := len(classMap)

	gnb.classlabels = make([]int, 0, classCount)
	gnb.priors = make([]float64, classCount)
	gnb.means = make([][]float64, classCount)
	gnb.variances = make([][]float64, classCount)

	sampleCount := len(X)
	featureCount := len(X[0])

	classIdx := 0
	for label, indices := range classMap {
		gnb.classlabels = append(gnb.classlabels, label)

		n := float64(len(indices))

		gnb.priors[classIdx] = n / float64(sampleCount)

		mean := make([]float64, featureCount)
		variance := make([]float64, featureCount)

		for _, idx := range indices {
			for i := range featureCount {
				mean[i] += X[idx][i]
			}
		}
		for i := range featureCount {
			mean[i] /= n
		}

		for _, idx := range indices {
			for i := range featureCount {
				d := X[idx][i] - mean[i]
				variance[i] += d * d
			}
		}
		for i := range featureCount {
			variance[i] = (variance[i] / n) + 1e-9
		}

		gnb.means[classIdx] = mean
		gnb.variances[classIdx] = variance

		classIdx++
	}

	gnb.features = x.Columns()

	gnb.featureIndexMap = make(map[string]int)
	for i, name := range gnb.features {
		gnb.featureIndexMap[name] = i
	}

	gnb.fitted = true

	return nil
}

// Predict estimates the most probable class for each observation in the input table.
//
// The method implements the Gaussian Probability Density Function (PDF) in the log-domain to ensure numerical stability and prevent underflow when multiplying small probabilities.
// For each class, it calculates the posterior probability:  log(P(C|X)) ∝ log(P(C)) + Σ log(P(xi|C)).
//
// It returns a table with a single column "y_pred" containing the predicted class labels.
func (gnb *GaussianNaiveBayes) Predict(x *rowan.Table) (*rowan.Table, error) {
	if x == nil {
		return nil, errors.New("x is nil")
	}

	if !gnb.fitted {
		return nil, errors.New("model is not fitted yet")
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return nil, err
	}

	sampleCount := len(X)
	classCount := len(gnb.classlabels)

	yPred := make([]any, sampleCount)

	logSquareRoot2Pi := math.Log(math.Sqrt(2 * math.Pi))

	for i := range sampleCount {
		bestClassIdx := -1
		maxLogProb := -math.MaxFloat64

		for c := range classCount {
			logProb := math.Log(gnb.priors[c])

			for j := range len(X[i]) {
				mean := gnb.means[c][j]
				variance := gnb.variances[c][j]

				x := X[i][j]

				e := -math.Pow(x-mean, 2) / (2 * variance)
				logLikelihood := -logSquareRoot2Pi - (0.5 * math.Log(variance)) + e

				logProb += logLikelihood
			}

			if logProb > maxLogProb {
				maxLogProb = logProb
				bestClassIdx = c
			}
		}

		yPred[i] = float64(gnb.classlabels[bestClassIdx])
	}

	return table.New(map[string][]any{"y_pred": yPred})
}

// IsFitted returns true if the model has been successfully trained.
func (gnb *GaussianNaiveBayes) IsFitted() bool {
	return gnb.fitted
}

// Features returns the names of the features the model was trained on.
func (gnb *GaussianNaiveBayes) Features() []string {
	features := make([]string, len(gnb.features))
	copy(features, gnb.features)

	return features
}

// FeatureIndex retrieves the numerical index of a feature by its name.
//
// It returns the index and a boolean indicating whether the feature exists.
func (gnb *GaussianNaiveBayes) FeatureIndex(feature string) (int, bool) {
	idx, ok := gnb.featureIndexMap[feature]

	return idx, ok
}

// Priors returns a deep copy of the class prior probabilities P(C).
//
// Each value represents the proportion of a specific class relative to the total number of samples in the training set. It returns an empty slice if the model has not been fitted.
func (gnb *GaussianNaiveBayes) Priors() []float64 {
	if !gnb.fitted {
		return make([]float64, 0)
	}

	priors := make([]float64, len(gnb.priors))
	copy(priors, gnb.priors)

	return priors
}

// PriorAt returns the prior probability P(C) of a specific class by its index.
//
// It returns an error if the model is not fitted or if the index is out of bounds.
func (gnb *GaussianNaiveBayes) PriorAt(classIdx int) (float64, error) {
	if !gnb.fitted {
		return 0, errors.New("model is not fitted")
	}

	if classIdx < 0 || classIdx >= len(gnb.priors) {
		return 0, errors.New("class index out of bounds")
	}

	return gnb.priors[classIdx], nil
}

// Means returns a deep copy of the entire mean matrix for all classes and features.
//
// Returns an empty matrix if the model has not been fitted.
func (gnb *GaussianNaiveBayes) Means() [][]float64 {
	if !gnb.fitted {
		return make([][]float64, 0)
	}

	return mathx.CopyMatrix(gnb.means)
}

// Variances returns a deep copy of the entire variance matrix for all classes and features.
//
// Returns an empty matrix if the model has not been fitted.
func (gnb *GaussianNaiveBayes) Variances() [][]float64 {
	if !gnb.fitted {
		return make([][]float64, 0)
	}

	return mathx.CopyMatrix(gnb.variances)
}

// MeanAt retrieves the mean value for a specific class and feature index.
//
// It returns an error if the model is not fitted or if the indices are out of bounds.
func (gnb *GaussianNaiveBayes) MeanAt(classIdx, featureIdx int) (float64, error) {
	if !gnb.fitted {
		return 0, errors.New("model is not fitted")
	}

	if classIdx < 0 || classIdx >= len(gnb.means) || featureIdx < 0 || featureIdx >= len(gnb.means[0]) {
		return 0, errors.New("index out of bounds")
	}

	return gnb.means[classIdx][featureIdx], nil
}

// VarianceAt retrieves the variance value for a specific class and feature index.
//
// It returns an error if the model is not fitted or if the indices are out of bounds.
func (gnb *GaussianNaiveBayes) VarianceAt(classIdx, featureIdx int) (float64, error) {
	if !gnb.fitted {
		return 0, errors.New("model is not fitted")
	}

	if classIdx < 0 || classIdx >= len(gnb.variances) || featureIdx < 0 || featureIdx >= len(gnb.variances[0]) {
		return 0, errors.New("index out of bounds")
	}

	return gnb.variances[classIdx][featureIdx], nil
}

// MeansByClass returns a deep copy of the mean vector for a specific class.
//
// This is useful for inspecting the expected values of features within a single category.
func (gnb *GaussianNaiveBayes) MeansByClass(classIdx int) ([]float64, error) {
	if !gnb.fitted {
		return nil, errors.New("model is not fitted")
	}

	if classIdx < 0 || classIdx >= len(gnb.means) {
		return nil, errors.New("class index out of bounds")
	}

	means := make([]float64, len(gnb.means[classIdx]))
	copy(means, gnb.means[classIdx])

	return means, nil
}

// VariancesByClass returns a deep copy of the variance vector for a specific class.
//
// This allows for the inspection of feature dispersion within a single category.
func (gnb *GaussianNaiveBayes) VariancesByClass(classIdx int) ([]float64, error) {
	if !gnb.fitted {
		return nil, errors.New("model is not fitted")
	}

	if classIdx < 0 || classIdx >= len(gnb.variances) {
		return nil, errors.New("class index out of bounds")
	}

	variances := make([]float64, len(gnb.variances[classIdx]))
	copy(variances, gnb.variances[classIdx])

	return variances, nil
}

// ClassLabels returns a copy of the unique target labels found in the training set.
//
// The order of labels in this slice corresponds to the class indices used internally for means, variances, and priors.
func (gnb *GaussianNaiveBayes) ClassLabels() []int {
	if !gnb.fitted {
		return make([]int, 0)
	}

	labels := make([]int, len(gnb.classlabels))
	copy(labels, gnb.classlabels)

	return labels
}

// ClassIdx retrieves the internal numerical index for a specific class label.
//
// Since this model treats targets as integers, the label provided must match the original integer value found in the training set. It returns the index and 'true' if found, or 0 and 'false' otherwise.
func (gnb *GaussianNaiveBayes) ClassIndex(label int) (int, bool) {
	for i, classLabel := range gnb.classlabels {
		if label == classLabel {
			return i, true
		}
	}

	return 0, false
}

// ClassLabelAt returns the original label value for a given internal class index.
func (gnb *GaussianNaiveBayes) ClassLabelAt(classIdx int) (int, error) {
	if !gnb.fitted {
		return 0, errors.New("model is not fitted")
	}

	if classIdx < 0 || classIdx >= len(gnb.classlabels) {
		return 0, errors.New("class index out of bounds")
	}

	return gnb.classlabels[classIdx], nil
}
