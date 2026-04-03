package eval

// Distance defines the contract for distance metrics between two numerical vectors.
type Distance interface {
	Name() string
	Measure(a, b []float64) (float64, error)
}

var (
	Euclidean = EuclideanDistance{}
	Manhattan = ManhattanDistance{}
	Chebyshev = ChebyshevDistance{}
)
