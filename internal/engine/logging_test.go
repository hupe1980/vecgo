package engine

import (
	"bytes"
	"context"
	"log/slog"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/require"
)

func TestStructuredLogging(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	e, err := Open(t.TempDir(), 128, distance.MetricL2, WithLogger(logger))
	require.NoError(t, err)
	defer e.Close()

	// Trigger a log event by flushing.
	// Insert some data to make flush do something.
	_, err = e.Insert(context.Background(), make([]float32, 128), nil, nil)
	require.NoError(t, err)

	err = e.Commit(context.Background())
	require.NoError(t, err)

	// Check logs
	logOutput := buf.String()
	require.Contains(t, logOutput, "Flush started")
	require.Contains(t, logOutput, "Flush completed")
	require.Contains(t, logOutput, `"rowCount":1`)
}
