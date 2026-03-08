package neighbors

import (
	"errors"
	"fmt"

	"github.com/go-rowan/rowan"
	"github.com/go-rowan/rowan-ml/preprocess"
)

func fit(x, y *rowan.Table, t preprocess.Transformer) ([][]float64, []float64, error) {
	if x == nil || y == nil {
		return nil, nil, errors.New("x and y must not be nil")
	}

	if x.Len() != y.Len() {
		return nil, nil, fmt.Errorf("features and target tables must have the same number of rows (%d vs %d)", x.Len(), y.Len())
	}

	if t != nil {
		err := t.Fit(x, x.Columns()...)
		if err != nil {
			return nil, nil, fmt.Errorf("scaling error during fit: %w", err)
		}

		x, err = t.Transform(x)
		if err != nil {
			return nil, nil, fmt.Errorf("scaling error during transform: %w", err)
		}
	}

	X, err := x.NumericMatrix()
	if err != nil {
		return nil, nil, err
	}

	Y, err := y.NumericSlice(0)
	if err != nil {
		return nil, nil, err
	}

	return X, Y, nil
}
