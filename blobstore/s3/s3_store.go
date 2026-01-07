package s3

import (
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

// Store implements blobstore.BlobStore for S3.
type Store struct {
	client *s3.Client
	bucket string
	prefix string
}

// NewStore creates a new S3 blob store.
// rootPrefix is prepended to all keys (e.g. "my-db/").
func NewStore(client *s3.Client, bucket, rootPrefix string) *Store {
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
	key := s.key(name)

	// Get metadata to verify existence and size
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

	return &s3Blob{
		client: s.client,
		bucket: s.bucket,
		key:    key,
		size:   *head.ContentLength,
	}, nil
}

func (s *Store) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	key := s.key(name)
	pr, pw := io.Pipe()

	blob := &s3WritableBlob{
		pw:       pw,
		done:     make(chan error, 1),
		uploader: manager.NewUploader(s.client),
	}

	// Start upload in background
	go func() {
		_, err := blob.uploader.Upload(context.Background(), &s3.PutObjectInput{
			Bucket: aws.String(s.bucket),
			Key:    aws.String(key),
			Body:   pr,
		})
		// Close the reader end of the pipe after upload completes/fails
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

// s3Blob implements blobstore.Blob
type s3Blob struct {
	client *s3.Client
	bucket string
	key    string
	size   int64
}

func (b *s3Blob) Close() error {
	return nil
}

func (b *s3Blob) Size() int64 {
	return b.size
}

func (b *s3Blob) ReadAt(p []byte, off int64) (int, error) {
	if off >= b.size {
		return 0, io.EOF
	}

	end := off + int64(len(p)) - 1
	if end >= b.size {
		end = b.size - 1
	}

	rangeHeader := fmt.Sprintf("bytes=%d-%d", off, end)

	resp, err := b.client.GetObject(context.Background(), &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(b.key),
		Range:  aws.String(rangeHeader),
	})
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	n, err := io.ReadFull(resp.Body, p)
	if err == io.ErrUnexpectedEOF {
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

func (b *s3Blob) ReadRange(off, lenReq int64) (io.ReadCloser, error) {
	if off >= b.size {
		return nil, io.EOF
	}

	end := off + lenReq - 1
	if end >= b.size {
		end = b.size - 1
	}

	rangeHeader := fmt.Sprintf("bytes=%d-%d", off, end)

	resp, err := b.client.GetObject(context.Background(), &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(b.key),
		Range:  aws.String(rangeHeader),
	})
	if err != nil {
		return nil, err
	}

	return resp.Body, nil
}

// s3WritableBlob implements blobstore.WritableBlob
type s3WritableBlob struct {
	pw       *io.PipeWriter
	done     chan error
	uploader *manager.Uploader
	closed   atomic.Bool
	writeErr error
}

func (b *s3WritableBlob) Write(p []byte) (int, error) {
	if b.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	return b.pw.Write(p)
}

func (b *s3WritableBlob) Close() error {
	if !b.closed.CompareAndSwap(false, true) {
		return io.ErrClosedPipe
	}
	if err := b.pw.Close(); err != nil {
		return err
	}
	err := <-b.done
	return err
}

func (b *s3WritableBlob) Sync() error {
	return nil
}
