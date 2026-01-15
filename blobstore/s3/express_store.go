package s3

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"path"
	"sort"
	"sync/atomic"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/hupe1980/vecgo/blobstore"
)

// ExpressStore implements blobstore.BlobStore for S3 Express One Zone.
//
// S3 Express One Zone is a high-performance, single-Availability Zone storage class
// that delivers consistent single-digit millisecond data access for frequently
// accessed data and latency-sensitive applications.
//
// Key differences from standard S3:
//   - Uses directory buckets (bucket names end with --azid--x-s3)
//   - Requires CreateSession for authentication
//   - Supports conditional writes (If-None-Match) for atomic operations
//   - Lower latency, higher throughput for compatible workloads
//
// Use this store for:
//   - Lambda functions requiring low-latency vector search
//   - Kubernetes workloads with ephemeral storage
//   - Real-time inference pipelines
type ExpressStore struct {
	client Client
	bucket string
	prefix string
}

// NewExpressStore creates a new S3 Express One Zone blob store.
// The bucket must be a directory bucket (ending with --azid--x-s3).
func NewExpressStore(client Client, bucket, rootPrefix string) *ExpressStore {
	return &ExpressStore{
		client: client,
		bucket: bucket,
		prefix: rootPrefix,
	}
}

func (s *ExpressStore) key(name string) string {
	return path.Join(s.prefix, name)
}

func (s *ExpressStore) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	key := s.key(name)

	head, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		var nf *types.NotFound
		if errors.As(err, &nf) {
			return nil, blobstore.ErrNotFound
		}
		var nsk *types.NoSuchKey
		if errors.As(err, &nsk) {
			return nil, blobstore.ErrNotFound
		}
		return nil, err
	}

	return &expressBlob{
		client: s.client,
		bucket: s.bucket,
		key:    key,
		size:   *head.ContentLength,
		etag:   aws.ToString(head.ETag),
	}, nil
}

// Put writes a blob atomically.
// S3 Express supports conditional writes via If-None-Match for true atomicity.
func (s *ExpressStore) Put(ctx context.Context, name string, data []byte) error {
	key := s.key(name)
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
		Body:   bytes.NewReader(data),
	})
	return err
}

// PutIfNotExists writes a blob only if it doesn't already exist.
// Uses S3 Express conditional writes for atomic create operations.
// Returns ErrConflict if the key already exists.
func (s *ExpressStore) PutIfNotExists(ctx context.Context, name string, data []byte) error {
	key := s.key(name)
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(s.bucket),
		Key:         aws.String(key),
		Body:        bytes.NewReader(data),
		IfNoneMatch: aws.String("*"), // Only succeed if object doesn't exist
	})
	if err != nil {
		// Check for precondition failed
		var apiErr interface{ ErrorCode() string }
		if errors.As(err, &apiErr) && apiErr.ErrorCode() == "PreconditionFailed" {
			return ErrConflict
		}
		return err
	}
	return nil
}

// ErrConflict is returned when a conditional write fails due to the object already existing.
var ErrConflict = errors.New("object already exists")

func (s *ExpressStore) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	key := s.key(name)
	pr, pw := io.Pipe()

	blob := &expressWritableBlob{
		pw:       pw,
		done:     make(chan error, 1),
		uploader: manager.NewUploader(s.client),
	}

	go func() {
		_, err := blob.uploader.Upload(ctx, &s3.PutObjectInput{
			Bucket: aws.String(s.bucket),
			Key:    aws.String(key),
			Body:   pr,
		})
		_ = pr.CloseWithError(err)
		blob.done <- err
	}()

	return blob, nil
}

func (s *ExpressStore) Delete(ctx context.Context, name string) error {
	key := s.key(name)
	_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	return err
}

func (s *ExpressStore) List(ctx context.Context, prefix string) ([]string, error) {
	fullPrefix := s.key(prefix)
	var keys []string

	paginator := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(fullPrefix),
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, obj := range page.Contents {
			relPath := *obj.Key
			if len(s.prefix) > 0 {
				if len(relPath) > len(s.prefix) && relPath[:len(s.prefix)] == s.prefix {
					relPath = relPath[len(s.prefix):]
					if len(relPath) > 0 && relPath[0] == '/' {
						relPath = relPath[1:]
					}
				}
			}
			keys = append(keys, relPath)
		}
	}
	sort.Strings(keys)
	return keys, nil
}

// expressBlob implements blobstore.Blob for S3 Express
type expressBlob struct {
	client Client
	bucket string
	key    string
	size   int64
	etag   string
}

func (b *expressBlob) Close() error {
	return nil
}

func (b *expressBlob) Size() int64 {
	return b.size
}

func (b *expressBlob) ReadAt(ctx context.Context, p []byte, off int64) (int, error) {
	if off >= b.size {
		return 0, io.EOF
	}

	if err := ctx.Err(); err != nil {
		return 0, err
	}

	end := off + int64(len(p)) - 1
	if end >= b.size {
		end = b.size - 1
	}

	rangeHeader := fmt.Sprintf("bytes=%d-%d", off, end)

	resp, err := b.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(b.key),
		Range:  aws.String(rangeHeader),
	})
	if err != nil {
		return 0, err
	}
	defer func() { _ = resp.Body.Close() }()

	n, err := io.ReadFull(resp.Body, p)
	if errors.Is(err, io.ErrUnexpectedEOF) {
		if off+int64(n) == b.size {
			return n, nil
		}
		return n, io.EOF
	}

	expected := end - off + 1
	if int64(n) == expected && int64(n) < int64(len(p)) {
		return n, io.EOF
	}

	return n, err
}

func (b *expressBlob) ReadRange(ctx context.Context, off, lenReq int64) (io.ReadCloser, error) {
	if off >= b.size {
		return nil, io.EOF
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	end := off + lenReq - 1
	if end >= b.size {
		end = b.size - 1
	}

	rangeHeader := fmt.Sprintf("bytes=%d-%d", off, end)

	resp, err := b.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(b.key),
		Range:  aws.String(rangeHeader),
	})
	if err != nil {
		return nil, err
	}

	return resp.Body, nil
}

// expressWritableBlob implements blobstore.WritableBlob
type expressWritableBlob struct {
	pw       *io.PipeWriter
	done     chan error
	uploader *manager.Uploader
	closed   atomic.Bool
}

func (b *expressWritableBlob) Write(p []byte) (int, error) {
	if b.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	return b.pw.Write(p)
}

func (b *expressWritableBlob) Close() error {
	if !b.closed.CompareAndSwap(false, true) {
		return io.ErrClosedPipe
	}
	if err := b.pw.Close(); err != nil {
		return err
	}
	return <-b.done
}

func (b *expressWritableBlob) Sync() error {
	return nil
}
