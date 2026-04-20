package classification

type naiveBayesOptions struct {
	alpha float64
}

type NaiveBayesOption func(*naiveBayesOptions)

func WithAlpha(a float64) NaiveBayesOption {
	return func(o *naiveBayesOptions) {
		o.alpha = a
	}
}
