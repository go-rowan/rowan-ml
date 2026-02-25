package preprocess

import "github.com/go-rowan/rowan"

// Transformer defines the interface for data preprocessing components that can learn from data and apply transformations.
//
// Fit computes and stores any statistics or parameters required to perform the transformation on the specified columns.
//
// Transform applies the learned transformation to the specified columns and returns a new transformed Table without modifying the original input.
//
// Implementations should return an error if Transform is called before Fit, or if the specified columns are invalid.
type Transformer interface {
	Fit(t *rowan.Table, columns ...string) error
	Transform(t *rowan.Table, columns ...string) (*rowan.Table, error)
}
