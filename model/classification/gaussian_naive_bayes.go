package classification

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan/table"
)

// GaussianNaiveBayes implements the Naive Bayes algorithm for continuous features, assuming that the likelihood of the features follows a Gaussian (normal) distribution.
//
// It stores the statistical parameters for each class-feature pair, allowing for efficient probability estimation during inference.
type GaussianNaiveBayes struct {
	means       [][]float64
	variances   [][]float64
	priors      []float64
	classlabels []int
	fitted      bool
}

// NewGaussianNaiveBayes initializes an empty GaussianNaiveBayes model.
//
// The model must be trained using the Fit method before it can be used to perform predictions.
func NewGaussianNaiveBayes() *GaussianNaiveBayes {
	return &GaussianNaiveBayes{
		fitted: false,
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

	classMap := make(map[int][]int)
	for i, val := range yData {
		classMap[val] = append(classMap[val], i)
	}

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
