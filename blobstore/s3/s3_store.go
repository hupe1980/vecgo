package s3

import (
	"bytes"
	"context"
	"io"
	"path"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/hupe1980/vecgo/blobstore"
)

// Store implements blobstore.BlobStore for S3.
type Store struct {
	client Client
	bucket string
	prefix string
}

// NewStore creates a new S3 blob store.
// rootPrefix is prepended to all keys (e.g. "my-db/").
func NewStore(client Client, bucket, rootPrefix string) *Store {
	return &Store{
		client: client,
		bucket: bucket,
		prefix: rootPrefix,
	}
}

func (s *Store) key(name string) string {
	return path.Join(s.prefix, name)
}

func (s *Store) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	return openBlob(ctx, s.client, s.bucket, s.key(name))
}

// Put writes a blob atomically.
func (s *Store) Put(ctx context.Context, name string, data []byte) error {
	key := s.key(name)
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
		Body:   bytes.NewReader(data),
	})
	return err
}

func (s *Store) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
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
