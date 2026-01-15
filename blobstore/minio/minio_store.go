package minio

import (
	"bytes"
	"context"
	"errors"
	"io"
	"path"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/minio/minio-go/v7"
)

// Store implements blobstore.BlobStore for MinIO and S3-compatible storage.
type Store struct {
	client *minio.Client
	bucket string
	prefix string
}

// NewStore creates a new MinIO blob store.
// bucket is the MinIO bucket name.
// rootPrefix is prepended to all keys (e.g. "vectors/").
func NewStore(client *minio.Client, bucket, rootPrefix string) *Store {
	return &Store{
		client: client,
		bucket: bucket,
		prefix: rootPrefix,
	}
}

func (s *Store) key(name string) string {
	return path.Join(s.prefix, name)
}

// Open opens an existing blob for reading.
func (s *Store) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	key := s.key(name)

	// Get object info to verify existence and get size
	info, err := s.client.StatObject(ctx, s.bucket, key, minio.StatObjectOptions{})
	if err != nil {
		errResp := minio.ToErrorResponse(err)
		if errResp.Code == "NoSuchKey" || errResp.Code == "NotFound" {
			return nil, blobstore.ErrNotFound
		}
		return nil, err
	}

	return &minioBlob{
		client: s.client,
		bucket: s.bucket,
		key:    key,
		size:   info.Size,
	}, nil
}

// Put writes a blob atomically.
func (s *Store) Put(ctx context.Context, name string, data []byte) error {
	key := s.key(name)
	_, err := s.client.PutObject(ctx, s.bucket, key, bytes.NewReader(data), int64(len(data)), minio.PutObjectOptions{})
	return err
}

// Create creates a new blob for streaming writes.
func (s *Store) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	key := s.key(name)
	pr, pw := io.Pipe()

	blob := &minioWritableBlob{
		pw:   pw,
		done: make(chan error, 1),
	}

	// Start upload in background
	go func() {
		_, err := s.client.PutObject(ctx, s.bucket, key, pr, -1, minio.PutObjectOptions{})
		_ = pr.CloseWithError(err)
		blob.done <- err
	}()

	return blob, nil
}

// Delete removes a blob.
func (s *Store) Delete(ctx context.Context, name string) error {
	key := s.key(name)
	err := s.client.RemoveObject(ctx, s.bucket, key, minio.RemoveObjectOptions{})
	if err != nil {
		errResp := minio.ToErrorResponse(err)
		if errResp.Code == "NoSuchKey" || errResp.Code == "NotFound" {
			return nil // Already gone
		}
		return err
	}
	return nil
}

// List returns all blob names with the given prefix.
func (s *Store) List(ctx context.Context, prefix string) ([]string, error) {
	fullPrefix := s.key(prefix)

	var names []string
	for obj := range s.client.ListObjects(ctx, s.bucket, minio.ListObjectsOptions{
		Prefix:    fullPrefix,
		Recursive: true,
	}) {
		if obj.Err != nil {
			return nil, obj.Err
		}
		// Strip our root prefix
		name := strings.TrimPrefix(obj.Key, s.prefix)
		name = strings.TrimPrefix(name, "/")
		if name != "" {
			names = append(names, name)
		}
	}

	sort.Strings(names)
	return names, nil
}

// minioBlob implements blobstore.Blob for MinIO.
type minioBlob struct {
	client *minio.Client
	bucket string
	key    string
	size   int64
}

func (b *minioBlob) Size() int64 {
	return b.size
}

func (b *minioBlob) ReadAt(ctx context.Context, p []byte, off int64) (int, error) {
	opts := minio.GetObjectOptions{}
	end := off + int64(len(p)) - 1
	if end >= b.size {
		end = b.size - 1
	}
	if err := opts.SetRange(off, end); err != nil {
		return 0, err
	}

	obj, err := b.client.GetObject(ctx, b.bucket, b.key, opts)
	if err != nil {
		return 0, err
	}
	defer obj.Close()

	return io.ReadFull(obj, p[:end-off+1])
}

func (b *minioBlob) ReadRange(ctx context.Context, off, length int64) (io.ReadCloser, error) {
	opts := minio.GetObjectOptions{}
	end := off + length - 1
	if end >= b.size {
		end = b.size - 1
	}
	if err := opts.SetRange(off, end); err != nil {
		return nil, err
	}

	obj, err := b.client.GetObject(ctx, b.bucket, b.key, opts)
	if err != nil {
		return nil, err
	}
	return obj, nil
}

func (b *minioBlob) Close() error {
	return nil
}

// minioWritableBlob implements blobstore.WritableBlob for MinIO.
type minioWritableBlob struct {
	pw       *io.PipeWriter
	done     chan error
	finished atomic.Bool
}

func (b *minioWritableBlob) Write(p []byte) (int, error) {
	return b.pw.Write(p)
}

func (b *minioWritableBlob) Close() error {
	if !b.finished.CompareAndSwap(false, true) {
		return errors.New("already closed")
	}
	if err := b.pw.Close(); err != nil {
		return err
	}
	return <-b.done
}

func (b *minioWritableBlob) Abort() error {
	if !b.finished.CompareAndSwap(false, true) {
		return nil
	}
	return b.pw.CloseWithError(errors.New("upload aborted"))
}

func (b *minioWritableBlob) Sync() error {
	return nil // Streaming upload, no sync needed
}
