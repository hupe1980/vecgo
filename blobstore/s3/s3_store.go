package s3

import (
	"bytes"
	"context"
	"path"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/hupe1980/vecgo/blobstore"
)

// Store implements blobstore.BlobStore for S3.
//
// Features:
//   - CRC32C checksum validation for data integrity
//   - Configurable multipart upload (part size, concurrency)
//   - Automatic abort of failed multipart uploads
//   - Production-optimized defaults
//
// For S3 Express One Zone (single-digit ms latency), use ExpressStore instead.
type Store struct {
	client   Client
	bucket   string
	prefix   string
	config   UploadConfig
	uploader *uploaderPool
}

// StoreOption configures a Store.
type StoreOption func(*Store)

// WithUploadConfig sets custom upload configuration.
func WithUploadConfig(cfg UploadConfig) StoreOption {
	return func(s *Store) {
		s.config = cfg
	}
}

// NewStore creates a new S3 blob store with production defaults.
// rootPrefix is prepended to all keys (e.g. "my-db/").
func NewStore(client Client, bucket, rootPrefix string, opts ...StoreOption) *Store {
	s := &Store{
		client: client,
		bucket: bucket,
		prefix: rootPrefix,
		config: DefaultUploadConfig(),
	}

	for _, opt := range opts {
		opt(s)
	}

	s.uploader = newUploaderPool(client, s.config)

	return s
}

func (s *Store) key(name string) string {
	return path.Join(s.prefix, name)
}

func (s *Store) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	return openBlob(ctx, s.client, s.bucket, s.key(name))
}

// Put writes a blob atomically with CRC32C integrity validation.
// Optimized for small files (manifests, metadata).
// For large files, use Create() which uses multipart upload.
func (s *Store) Put(ctx context.Context, name string, data []byte) error {
	key := s.key(name)

	if s.config.EnableChecksum {
		return putWithChecksum(ctx, s.client, s.bucket, key, data)
	}

	// Fallback without checksum
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        aws.String(s.bucket),
		Key:           aws.String(key),
		Body:          bytes.NewReader(data),
		ContentLength: aws.Int64(int64(len(data))),
	})
	return err
}

// Create creates a streaming upload for large files.
// Uses multipart upload with configurable part size and concurrency.
// The upload is automatically aborted if it fails.
func (s *Store) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
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

func (s *Store) Delete(ctx context.Context, name string) error {
	key := s.key(name)
	_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	return err
}

func (s *Store) List(ctx context.Context, prefix string) ([]string, error) {
	return listObjects(ctx, s.client, s.bucket, s.key(prefix), s.prefix)
}

// AbortIncompleteUploads cleans up orphaned multipart uploads.
// Call this periodically (e.g., on startup or via cron) to reclaim storage.
//
// Alternatively, configure an S3 lifecycle rule to auto-abort incomplete uploads:
//
//	{
//	  "Rules": [{
//	    "ID": "AbortIncompleteMultipartUploads",
//	    "Status": "Enabled",
//	    "AbortIncompleteMultipartUpload": { "DaysAfterInitiation": 1 }
//	  }]
//	}
func (s *Store) AbortIncompleteUploads(ctx context.Context) error {
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
				// Log but continue - best effort cleanup
				continue
			}
		}
	}

	return nil
}

// uploaderPool manages a pool of uploaders for concurrent uploads.
// This avoids creating a new uploader per upload while maintaining
// isolation for concurrent operations.
type uploaderPool struct {
	client Client
	config UploadConfig
}

func newUploaderPool(client Client, config UploadConfig) *uploaderPool {
	return &uploaderPool{
		client: client,
		config: config,
	}
}

func (p *uploaderPool) get() *manager.Uploader {
	// Create a fresh uploader per upload to avoid any state sharing issues
	// The uploader is lightweight - the expensive part is the actual upload
	return newUploader(p.client, p.config)
}
