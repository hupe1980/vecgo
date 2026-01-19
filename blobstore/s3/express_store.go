package s3

import (
	"bytes"
	"context"
	"errors"
	"path"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/smithy-go"
	"github.com/hupe1980/vecgo/blobstore"
)

// ErrConflict is returned when a conditional write fails due to the object already existing.
var ErrConflict = errors.New("object already exists")

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
//
// Features:
//   - CRC32C checksum validation for data integrity
//   - Configurable multipart upload (part size, concurrency)
//   - Conditional writes (If-None-Match) for atomic create operations
//   - Automatic abort of failed multipart uploads
type ExpressStore struct {
	client   Client
	bucket   string
	prefix   string
	config   UploadConfig
	uploader *uploaderPool
}

// NewExpressStore creates a new S3 Express One Zone blob store.
// The bucket must be a directory bucket (ending with --azid--x-s3).
func NewExpressStore(client Client, bucket, rootPrefix string, opts ...StoreOption) *ExpressStore {
	// Create a temporary Store to apply options
	s := &Store{config: DefaultUploadConfig()}
	for _, opt := range opts {
		opt(s)
	}

	return &ExpressStore{
		client:   client,
		bucket:   bucket,
		prefix:   rootPrefix,
		config:   s.config,
		uploader: newUploaderPool(client, s.config),
	}
}

func (s *ExpressStore) key(name string) string {
	return path.Join(s.prefix, name)
}

func (s *ExpressStore) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	return openBlob(ctx, s.client, s.bucket, s.key(name))
}

// Put writes a blob atomically with CRC32C integrity validation.
func (s *ExpressStore) Put(ctx context.Context, name string, data []byte) error {
	key := s.key(name)

	if s.config.EnableChecksum {
		return putWithChecksum(ctx, s.client, s.bucket, key, data)
	}

	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        aws.String(s.bucket),
		Key:           aws.String(key),
		Body:          bytes.NewReader(data),
		ContentLength: aws.Int64(int64(len(data))),
	})
	return err
}

// PutIfNotExists writes a blob only if it doesn't already exist.
// Uses S3 Express conditional writes for atomic create operations.
// Returns ErrConflict if the key already exists.
func (s *ExpressStore) PutIfNotExists(ctx context.Context, name string, data []byte) error {
	key := s.key(name)

	input := &s3.PutObjectInput{
		Bucket:        aws.String(s.bucket),
		Key:           aws.String(key),
		Body:          bytes.NewReader(data),
		ContentLength: aws.Int64(int64(len(data))),
		IfNoneMatch:   aws.String("*"), // Only succeed if object doesn't exist
	}

	if s.config.EnableChecksum {
		input.ChecksumCRC32C = aws.String(computeCRC32C(data))
	}

	_, err := s.client.PutObject(ctx, input)
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

// Create creates a streaming upload for large files.
func (s *ExpressStore) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	key := s.key(name)
	uploader := s.uploader.get()

	return newStreamingWritableBlob(
		ctx,
		s.client,
		uploader,
		s.bucket,
		key,
		s.config.EnableChecksum,
	), nil
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

// AbortIncompleteUploads cleans up orphaned multipart uploads.
func (s *ExpressStore) AbortIncompleteUploads(ctx context.Context) error {
	paginator := s3.NewListMultipartUploadsPaginator(s.client, &s3.ListMultipartUploadsInput{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(s.prefix),
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return err
		}

		for _, upload := range page.Uploads {
			_, err := s.client.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
				Bucket:   aws.String(s.bucket),
				Key:      upload.Key,
				UploadId: upload.UploadId,
			})
			if err != nil {
				continue
			}
		}
	}

	return nil
}
