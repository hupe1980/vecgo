package s3

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/hupe1980/vecgo/blobstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntegration_S3Store(t *testing.T) {
	bucket := os.Getenv("S3_BUCKET")
	if bucket == "" {
		t.Skip("Skipping S3 integration test: S3_BUCKET not set")
	}

	ctx := context.Background()
	cfg, err := config.LoadDefaultConfig(ctx)
	require.NoError(t, err)

	client := s3.NewFromConfig(cfg)

	// Create a unique prefix for this test run
	prefix := fmt.Sprintf("test-vecgo-%d/", time.Now().UnixNano())
	store := NewStore(client, bucket, prefix)

	t.Run("Create and Read", func(t *testing.T) {
		name := "test.blob"
		data := make([]byte, 1024*1024) // 1MB
		rand.Read(data)

		// Create
		w, err := store.Create(ctx, name)
		require.NoError(t, err)
		n, err := w.Write(data)
		require.NoError(t, err)
		assert.Equal(t, len(data), n)
		require.NoError(t, w.Close())

		// List
		blobs, err := store.List(ctx, "")
		require.NoError(t, err)
		assert.Contains(t, blobs, name)

		// Open
		r, err := store.Open(ctx, name)
		require.NoError(t, err)
		assert.Equal(t, int64(len(data)), r.Size())

		// ReadAt
		buf := make([]byte, 100)
		n2, err := r.ReadAt(ctx, buf, 0)
		require.NoError(t, err)
		assert.Equal(t, 100, n2)
		assert.Equal(t, data[:100], buf)

		// ReadAt Offset
		n3, err := r.ReadAt(ctx, buf, 1024)
		require.NoError(t, err)
		assert.Equal(t, 100, n3)
		assert.Equal(t, data[1024:1124], buf)

		// Clean up
		require.NoError(t, store.Delete(ctx, name))
		require.NoError(t, r.Close())
	})

	t.Run("NotFound", func(t *testing.T) {
		_, err := store.Open(ctx, "nonexistent")
		assert.ErrorIs(t, err, blobstore.ErrNotFound)
	})
}
