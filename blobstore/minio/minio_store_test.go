package minio

import (
	"context"
	"testing"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMinioStore_Integration requires a running MinIO instance.
// Skip if not available.
func TestMinioStore_Integration(t *testing.T) {
	endpoint := "localhost:9000"
	accessKey := "minioadmin"
	secretKey := "minioadmin"
	bucket := "test-vecgo"

	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKey, secretKey, ""),
		Secure: false,
	})
	if err != nil {
		t.Skipf("MinIO client creation failed: %v", err)
	}

	ctx := context.Background()

	// Check if MinIO is reachable
	_, err = client.ListBuckets(ctx)
	if err != nil {
		t.Skipf("MinIO not available: %v", err)
	}

	// Ensure bucket exists
	exists, err := client.BucketExists(ctx, bucket)
	require.NoError(t, err)
	if !exists {
		err = client.MakeBucket(ctx, bucket, minio.MakeBucketOptions{})
		require.NoError(t, err)
	}

	store := NewStore(client, bucket, "test-prefix/")

	// Test Put and Open
	data := []byte("hello minio world")
	err = store.Put(ctx, "test.txt", data)
	require.NoError(t, err)

	blob, err := store.Open(ctx, "test.txt")
	require.NoError(t, err)
	require.Equal(t, int64(len(data)), blob.Size())

	buf := make([]byte, len(data))
	n, err := blob.ReadAt(ctx, buf, 0)
	require.NoError(t, err)
	require.Equal(t, len(data), n)
	require.Equal(t, data, buf)
	require.NoError(t, blob.Close())

	// Test ReadRange
	blob2, err := store.Open(ctx, "test.txt")
	require.NoError(t, err)
	rc, err := blob2.ReadRange(ctx, 6, 5)
	require.NoError(t, err)
	partBuf := make([]byte, 5)
	_, err = rc.Read(partBuf)
	require.NoError(t, err)
	assert.Equal(t, "minio", string(partBuf))
	require.NoError(t, rc.Close())
	require.NoError(t, blob2.Close())

	// Test List
	names, err := store.List(ctx, "")
	require.NoError(t, err)
	assert.Contains(t, names, "test.txt")

	// Test Delete
	err = store.Delete(ctx, "test.txt")
	require.NoError(t, err)

	// Verify deleted
	_, err = store.Open(ctx, "test.txt")
	require.Error(t, err)

	// Test Create (streaming)
	wb, err := store.Create(ctx, "stream.txt")
	require.NoError(t, err)
	_, err = wb.Write([]byte("streamed data"))
	require.NoError(t, err)
	err = wb.Close()
	require.NoError(t, err)

	blob3, err := store.Open(ctx, "stream.txt")
	require.NoError(t, err)
	assert.Equal(t, int64(13), blob3.Size())
	require.NoError(t, blob3.Close())

	// Cleanup
	_ = store.Delete(ctx, "stream.txt")
}
