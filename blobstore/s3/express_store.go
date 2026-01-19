package s3

import (
	"bytes"
	"context"
	"errors"
	"io"
	"path"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/smithy-go"
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
	return openBlob(ctx, s.client, s.bucket, s.key(name))
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
		// Check for precondition failed (object already exists)
		// S3 Express returns PreconditionFailed or ConditionalRequestConflict
		var apiErr smithy.APIError
		if errors.As(err, &apiErr) {
			code := apiErr.ErrorCode()
			if code == "PreconditionFailed" || code == "ConditionalRequestConflict" {
				return ErrConflict
			}
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

	blob := &baseWritableBlob{
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
	return listObjects(ctx, s.client, s.bucket, s.key(prefix), s.prefix)
}
