package eval

import (
	"fmt"

	"github.com/go-rowan/rowan/table"
)

// Report represents a collection of metric names and their calculated scores.
type Report map[string]float64

// Eval evaluates model performance by computing multiple metrics at once.
// It returns a Report containing the results or an error if any metric computation fails.
func Eval(yTrue, yPredict []float64, metrics ...Metric) (Report, error) {
	report := make(Report, len(metrics))

	for _, m := range metrics {
		name := m.Name()

		score, err := m.Compute(yTrue, yPredict)
		if err != nil {
			return nil, fmt.Errorf("failed performing %s: %v", name, err)
		}

		report[name] = score
	}

	return report, nil
}

// GetScore returns the score for a specific metric name.
// It returns an error if the metric name does not exist in the report.
func (r Report) GetScore(metricName string) (float64, error) {
	score, ok := r[metricName]
	if !ok {
		return 0, fmt.Errorf("failed fetching value from key %s", metricName)
	}

	return score, nil
}

// MustGetScore returns the score for a specific metric name without error checking.
// Use this only when you are certain the metric exists in the report.
func (r Report) MustGetScore(metricName string) float64 {
	return r[metricName]
}

// Metrics returns a slice of all metric names present in the report.
func (r Report) Metrics() []string {
	metrics := make([]string, 0, len(r))

	for key := range r {
		metrics = append(metrics, key)
	}

	return metrics
}

// ShowAll renders all metrics contained in the Report in a transposed tabular format.
//
// The report is internally converted into a temporary table and displayed using DisplayTranspose.
// This method is primarily intended for interactive usage, CLI output, or quick inspection of evaluation results.
//
// ShowAll returns an error if the temporary table cannot be created.
func (r Report) ShowAll() error {
	data := make(map[string][]any, len(r))

	for key, value := range r {
		data[key] = []any{value}
	}

	tbl, err := table.New(data, NameRegressionMetrics())
	if err != nil {
		return fmt.Errorf("show all: failed creating temporary table: %w", err)
	}

	tbl.DisplayTranspose()
	return nil
}
