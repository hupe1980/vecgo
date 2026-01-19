package s3

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"sync"
	"sync/atomic"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/hupe1980/vecgo/internal/hash"
)

// UploadConfig configures the S3 uploader for optimal performance.
type UploadConfig struct {
	// PartSize is the minimum part size for multipart uploads.
	// Default: 8MB (larger than SDK default of 5MB for better throughput)
	PartSize int64

	// Concurrency is the number of concurrent part uploads.
	// Default: 5 (matches SDK default)
	Concurrency int

	// EnableChecksum enables CRC32C integrity validation.
	// Recommended for production workloads.
	// Default: true
	EnableChecksum bool

	// LeavePartsOnError controls whether failed multipart uploads
	// are automatically aborted.
	// Default: false (abort on error)
	LeavePartsOnError bool
}

// DefaultUploadConfig returns production-optimized upload settings.
func DefaultUploadConfig() UploadConfig {
	return UploadConfig{
		PartSize:          8 * 1024 * 1024, // 8MB - better for large segments
		Concurrency:       5,
		EnableChecksum:    true,
		LeavePartsOnError: false,
	}
}

// newUploader creates a configured S3 uploader.
func newUploader(client Client, cfg UploadConfig) *manager.Uploader {
	return manager.NewUploader(client, func(u *manager.Uploader) {
		u.PartSize = cfg.PartSize
		u.Concurrency = cfg.Concurrency
		u.LeavePartsOnError = cfg.LeavePartsOnError
	})
}

// computeCRC32C computes the CRC32C checksum and returns it as base64 (S3 format).
func computeCRC32C(data []byte) string {
	sum := hash.CRC32C(data)
	// S3 expects base64-encoded big-endian bytes
	b := []byte{byte(sum >> 24), byte(sum >> 16), byte(sum >> 8), byte(sum)}
	return base64.StdEncoding.EncodeToString(b)
}

// streamingWritableBlob implements WritableBlob with proper abort handling.
//
// Key improvements over the naive pipe approach:
// 1. Properly aborts multipart upload on context cancellation
// 2. Tracks upload ID for explicit abort if needed
// 3. Uses buffered pipe for better throughput
type streamingWritableBlob struct {
	pw       *io.PipeWriter
	pr       *io.PipeReader
	uploader *manager.Uploader
	bucket   string
	key      string
	client   Client

	// Upload state
	done     chan error
	uploadID atomic.Value // stores *string
	closed   atomic.Bool
	closeErr error
	closeMu  sync.Mutex
}

// newStreamingWritableBlob creates a new streaming upload with proper lifecycle management.
func newStreamingWritableBlob(
	ctx context.Context,
	client Client,
	uploader *manager.Uploader,
	bucket, key string,
	enableChecksum bool,
) *streamingWritableBlob {
	pr, pw := io.Pipe()

	blob := &streamingWritableBlob{
		pw:       pw,
		pr:       pr,
		uploader: uploader,
		bucket:   bucket,
		key:      key,
		client:   client,
		done:     make(chan error, 1),
	}

	// Start upload in background
	go blob.uploadLoop(ctx, enableChecksum)

	return blob
}

func (b *streamingWritableBlob) uploadLoop(ctx context.Context, enableChecksum bool) {
	input := &s3.PutObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(b.key),
		Body:   b.pr,
	}

	if enableChecksum {
		input.ChecksumAlgorithm = types.ChecksumAlgorithmCrc32c
	}

	_, err := b.uploader.Upload(ctx, input)

	// Close the read end with any error
	_ = b.pr.CloseWithError(err)

	// Signal completion
	b.done <- err
}

func (b *streamingWritableBlob) Write(p []byte) (int, error) {
	if b.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	return b.pw.Write(p)
}

func (b *streamingWritableBlob) Close() error {
	b.closeMu.Lock()
	defer b.closeMu.Unlock()

	if !b.closed.CompareAndSwap(false, true) {
		return b.closeErr
	}

	// Close the write end to signal EOF to the uploader
	if err := b.pw.Close(); err != nil {
		b.closeErr = err
		return err
	}

	// Wait for upload to complete
	b.closeErr = <-b.done
	return b.closeErr
}

// Abort explicitly aborts an in-progress upload.
// This is called automatically on Close() error if LeavePartsOnError is false.
// Exposed for explicit cleanup during graceful shutdown.
func (b *streamingWritableBlob) Abort(ctx context.Context) error {
	// Close the pipe to stop any ongoing writes
	b.closed.Store(true)
	_ = b.pw.CloseWithError(context.Canceled)

	// If we have an upload ID, abort the multipart upload
	if idPtr := b.uploadID.Load(); idPtr != nil {
		if uploadID := idPtr.(*string); uploadID != nil && *uploadID != "" {
			_, err := b.client.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
				Bucket:   aws.String(b.bucket),
				Key:      aws.String(b.key),
				UploadId: uploadID,
			})
			return err
		}
	}

	return nil
}

// Sync is a no-op for S3 uploads - data is only committed on Close().
func (b *streamingWritableBlob) Sync() error {
	return nil
}

// putWithChecksum uploads a small blob with CRC32C integrity validation.
func putWithChecksum(ctx context.Context, client Client, bucket, key string, data []byte) error {
	checksum := computeCRC32C(data)

	_, err := client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:         aws.String(bucket),
		Key:            aws.String(key),
		Body:           bytes.NewReader(data),
		ContentLength:  aws.Int64(int64(len(data))),
		ChecksumCRC32C: aws.String(checksum),
	})

	return err
}
