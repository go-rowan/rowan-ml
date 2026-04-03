package eval

import (
	"errors"
	"math"

	"github.com/go-rowan/rowan-ml/internal/slice"
)

// SilhouetteEvaluator implements the Silhouette Coefficient algorithm, a validation technique for assessing the quality of clustering results.
type SilhouetteEvaluator struct {
	distance Distance
}

// NewSilhouette initializes a SilhouetteEvaluator with a specified distance metric.
//
// Defaults to Euclidean distance if no metric is provided.
func NewSilhouette(distance ...Distance) *SilhouetteEvaluator {
	var d Distance
	d = Euclidean

	if len(distance) > 0 {
		d = distance[0]
	}

	return &SilhouetteEvaluator{
		distance: d,
	}
}

// Compute calculates the mean Silhouette Coefficient for all samples in the dataset.
//
// The coefficient ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. It requires a feature matrix X and their corresponding cluster labels.
func (s *SilhouetteEvaluator) Compute(X [][]float64, labels []int) (float64, error) {
	n := len(X)
	if n == 0 || len(labels) != n {
		return 0, errors.New("invalid length")
	}

	var total float64

	for i := 0; i < n; i++ {
		a, err := s.a(i, X, labels)
		if err != nil {
			return 0, err
		}

		b, err := s.b(i, X, labels)
		if err != nil {
			return 0, err
		}

		max := math.Max(a, b)
		if max > 0 {
			total += (b - a) / max
		}
	}

	return total / float64(n), nil
}

func (s *SilhouetteEvaluator) a(i int, X [][]float64, labels []int) (float64, error) {
	lbl := labels[i]

	var (
		sum   float64
		count int
	)

	for j := 0; j < len(X); j++ {
		if i != j && labels[j] == lbl {
			m, err := s.distance.Measure(X[i], X[j])
			if err != nil {
				return 0, err
			}

			sum += m
			count++
		}
	}

	if count == 0 {
		return 0, nil
	}

	return sum / float64(count), nil
}

func (s *SilhouetteEvaluator) b(i int, X [][]float64, labels []int) (float64, error) {
	lbl := labels[i]
	min := math.MaxFloat64

	unique := slice.UniqueInts(labels)

	for _, label := range unique {
		if label == lbl {
			continue
		}

		var (
			sum   float64
			count int
		)

		for j := 0; j < len(X); j++ {
			if labels[j] == label {
				m, err := s.distance.Measure(X[i], X[j])
				if err != nil {
					return 0, err
				}

				sum += m
				count++
			}
		}

		if count > 0 {
			avg := sum / float64(count)

			if avg < min {
				min = avg
			}
		}
	}

	return min, nil
}
