package eval

// Distance defines the contract for distance metrics between two numerical vectors.
type Distance interface {
	Name() string
	Compute(a, b []float64) (float64, error)
}

var (
	Euclidean = EuclideanDistance{}
)
