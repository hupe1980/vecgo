package flat

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChecksum(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checksum.bin")
	st := blobstore.NewLocalStore(dir)

	// 1. Write valid segment
	f, err := os.Create(path)
	require.NoError(t, err)

	w := NewWriter(f, nil, 1, 2, distance.MetricL2, 0, QuantizationNone)
	err = w.Add(model.ID(1), []float32{1.0, 0.0}, nil, nil)
	require.NoError(t, err)
	err = w.Flush()
	require.NoError(t, err)
	f.Close()

	// 2. Open with verification (should pass)
	blob, err := st.Open(context.Background(), "checksum.bin")
	require.NoError(t, err)
	seg, err := Open(blob, WithVerifyChecksum(true))
	require.NoError(t, err)
	seg.Close()

	// 3. Corrupt the file
	data, err := os.ReadFile(path)
	require.NoError(t, err)

	// Flip a bit in the body (e.g. in the vector data)
	// Header is first ~100 bytes.
	// We have 1 vector (8 bytes).
	// Corrupt byte at HeaderSize + 1
	if len(data) > HeaderSize+1 {
		data[HeaderSize+1] ^= 0xFF
	} else {
		t.Fatal("File too small to corrupt body")
	}

	err = os.WriteFile(path, data, 0644)
	require.NoError(t, err)

	// 4. Open with verification (should fail)
	blob, err = st.Open(context.Background(), "checksum.bin")
	require.NoError(t, err)
	_, err = Open(blob, WithVerifyChecksum(true))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "checksum mismatch")

	// 5. Open without verification (should pass, but might have bad data)
	blob, err = st.Open(context.Background(), "checksum.bin")
	require.NoError(t, err)
	seg2, err := Open(blob, WithVerifyChecksum(false))
	require.NoError(t, err)
	seg2.Close()
}
