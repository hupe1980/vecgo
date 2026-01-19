package s3

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"sync/atomic"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/hupe1980/vecgo/blobstore"
)

// baseBlob provides common implementation for S3 blob operations.
// Both s3Blob and expressBlob embed this for shared functionality.
type baseBlob struct {
	client Client
	bucket string
	key    string
	size   int64
}

func (b *baseBlob) Close() error {
	return nil
}

func (b *baseBlob) Size() int64 {
	return b.size
}

// ReadAt reads len(p) bytes starting at offset off.
// Implements blobstore.Blob interface.
func (b *baseBlob) ReadAt(ctx context.Context, p []byte, off int64) (int, error) {
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

// ReadRange returns a reader for a range of bytes.
// Implements blobstore.Blob interface.
func (b *baseBlob) ReadRange(ctx context.Context, off, lenReq int64) (io.ReadCloser, error) {
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

// listObjects is a shared helper for listing S3 objects.
func listObjects(ctx context.Context, client Client, bucket, fullPrefix, rootPrefix string) ([]string, error) {
	var keys []string

	paginator := s3.NewListObjectsV2Paginator(client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(fullPrefix),
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, obj := range page.Contents {
			relPath := *obj.Key
			if len(rootPrefix) > 0 {
				if len(relPath) > len(rootPrefix) && relPath[:len(rootPrefix)] == rootPrefix {
					relPath = relPath[len(rootPrefix):]
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

// openBlob is a shared helper for opening S3 blobs.
func openBlob(ctx context.Context, client Client, bucket, key string) (*baseBlob, error) {
	head, err := client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(bucket),
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

	return &baseBlob{
		client: client,
		bucket: bucket,
		key:    key,
		size:   *head.ContentLength,
	}, nil
}

// baseWritableBlob provides common implementation for writable S3 blobs.
type baseWritableBlob struct {
	pw       *io.PipeWriter
	done     chan error
	uploader *manager.Uploader
	closed   atomic.Bool
}

func (b *baseWritableBlob) Write(p []byte) (int, error) {
	if b.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	return b.pw.Write(p)
}

func (b *baseWritableBlob) Close() error {
	if !b.closed.CompareAndSwap(false, true) {
		return io.ErrClosedPipe
	}
	if err := b.pw.Close(); err != nil {
		return err
	}
	return <-b.done
}

// Sync is a no-op for S3 uploads.
// The upload is only finalized when Close() is called.
func (b *baseWritableBlob) Sync() error {
	return nil
}
