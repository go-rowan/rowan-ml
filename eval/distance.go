package eval

type Distance interface {
	Name() string
	Compute(a, b []float64) (float64, error)
}
