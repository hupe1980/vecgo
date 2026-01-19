package s3

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
	"github.com/hupe1980/vecgo/blobstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestStore_Open(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewStore(mockClient, "test-bucket", "prefix")

	t.Run("NotFound", func(t *testing.T) {
		mockClient.On("HeadObject", mock.Anything, mock.MatchedBy(func(input *s3.HeadObjectInput) bool {
			return *input.Bucket == "test-bucket" && *input.Key == "prefix/foo"
		})).Return(nil, &types.NotFound{}).Once()

		_, err := store.Open(context.Background(), "foo")
		assert.Equal(t, blobstore.ErrNotFound, err)
	})

	t.Run("Success", func(t *testing.T) {
		mockClient.On("HeadObject", mock.Anything, mock.MatchedBy(func(input *s3.HeadObjectInput) bool {
			return *input.Bucket == "test-bucket" && *input.Key == "prefix/bar"
		})).Return(&s3.HeadObjectOutput{
			ContentLength: aws.Int64(100),
		}, nil).Once()

		blob, err := store.Open(context.Background(), "bar")
		assert.NoError(t, err)
		assert.Equal(t, int64(100), blob.Size())
	})
}

func TestStore_Delete(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewStore(mockClient, "test-bucket", "prefix")

	mockClient.On("DeleteObject", mock.Anything, mock.MatchedBy(func(input *s3.DeleteObjectInput) bool {
		return *input.Bucket == "test-bucket" && *input.Key == "prefix/del"
	})).Return(&s3.DeleteObjectOutput{}, nil).Once()

	err := store.Delete(context.Background(), "del")
	assert.NoError(t, err)
}

func TestStore_List(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewStore(mockClient, "test-bucket", "prefix/")

	mockClient.On("ListObjectsV2", mock.Anything, mock.MatchedBy(func(input *s3.ListObjectsV2Input) bool {
		return *input.Bucket == "test-bucket" && *input.Prefix == "prefix"
	})).Return(&s3.ListObjectsV2Output{
		Contents: []types.Object{
			{Key: aws.String("prefix/file1")},
			{Key: aws.String("prefix/dir/file2")},
		},
	}, nil).Once()

	keys, err := store.List(context.Background(), "")
	assert.NoError(t, err)
	assert.Equal(t, []string{"dir/file2", "file1"}, keys)
}

func TestStore_List_Pagination(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewStore(mockClient, "test-bucket", "prefix/")

	// Page 1
	mockClient.On("ListObjectsV2", mock.Anything, mock.MatchedBy(func(input *s3.ListObjectsV2Input) bool {
		return input.ContinuationToken == nil
	})).Return(&s3.ListObjectsV2Output{
		IsTruncated:           aws.Bool(true),
		NextContinuationToken: aws.String("token"),
		Contents:              []types.Object{{Key: aws.String("prefix/1")}},
	}, nil).Once()

	// Page 2
	mockClient.On("ListObjectsV2", mock.Anything, mock.MatchedBy(func(input *s3.ListObjectsV2Input) bool {
		return input.ContinuationToken != nil && *input.ContinuationToken == "token"
	})).Return(&s3.ListObjectsV2Output{
		IsTruncated: aws.Bool(false),
		Contents:    []types.Object{{Key: aws.String("prefix/2")}},
	}, nil).Once()

	keys, err := store.List(context.Background(), "")
	assert.NoError(t, err)
	assert.Equal(t, []string{"1", "2"}, keys)
}

func TestBlob_ReadAt(t *testing.T) {
	mockClient := new(MockS3Client)
	blob := &baseBlob{
		client: mockClient,
		bucket: "b",
		key:    "k",
		size:   10,
	}

	mockClient.On("GetObject", mock.Anything, mock.MatchedBy(func(input *s3.GetObjectInput) bool {
		return *input.Bucket == "b" && *input.Key == "k" && *input.Range == "bytes=0-4"
	})).Return(&s3.GetObjectOutput{
		Body: io.NopCloser(strings.NewReader("hello")),
	}, nil).Once()

	ctx := context.Background()
	buf := make([]byte, 5)
	n, err := blob.ReadAt(ctx, buf, 0)
	assert.Equal(t, 5, n)
	assert.NoError(t, err)
	assert.Equal(t, "hello", string(buf))
}

func TestBlob_ReadRange(t *testing.T) {
	mockClient := new(MockS3Client)
	blob := &baseBlob{
		client: mockClient,
		bucket: "b",
		key:    "k",
		size:   10,
	}

	mockClient.On("GetObject", mock.Anything, mock.MatchedBy(func(input *s3.GetObjectInput) bool {
		return *input.Bucket == "b" && *input.Key == "k" && *input.Range == "bytes=2-6"
	})).Return(&s3.GetObjectOutput{
		Body: io.NopCloser(strings.NewReader("llo W")),
	}, nil).Once()

	ctx := context.Background()
	r, err := blob.ReadRange(ctx, 2, 5)
	assert.NoError(t, err)
	defer r.Close()

	buf, err := io.ReadAll(r)
	assert.NoError(t, err)
	assert.Equal(t, "llo W", string(buf))
}

func TestStore_Create(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewStore(mockClient, "test-bucket", "prefix")

	// Use Run/Return to consume the body asynchronously (safe-ish with pipe)
	// Note: manager.Uploader might buffer data, so PutObject might receive a buffer, not the pipe directly.
	mockClient.On("PutObject", mock.Anything, mock.MatchedBy(func(input *s3.PutObjectInput) bool {
		return *input.Bucket == "test-bucket" && *input.Key == "prefix/new"
	})).Run(func(args mock.Arguments) {
		input := args.Get(1).(*s3.PutObjectInput)
		// Consume body to let pipe finish
		io.ReadAll(input.Body)
	}).Return(&s3.PutObjectOutput{}, nil).Once()

	wb, err := store.Create(context.Background(), "new")
	assert.NoError(t, err)

	_, err = wb.Write([]byte("content"))
	assert.NoError(t, err)

	err = wb.Close()
	assert.NoError(t, err)
}

// ExpressStore tests

func TestExpressStore_Open(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

	t.Run("NotFound", func(t *testing.T) {
		mockClient.On("HeadObject", mock.Anything, mock.MatchedBy(func(input *s3.HeadObjectInput) bool {
			return *input.Bucket == "express-bucket--usw2-az1--x-s3" && *input.Key == "data/missing"
		})).Return(nil, &types.NotFound{}).Once()

		_, err := store.Open(context.Background(), "missing")
		assert.Equal(t, blobstore.ErrNotFound, err)
	})

	t.Run("Success", func(t *testing.T) {
		mockClient.On("HeadObject", mock.Anything, mock.MatchedBy(func(input *s3.HeadObjectInput) bool {
			return *input.Bucket == "express-bucket--usw2-az1--x-s3" && *input.Key == "data/segment.bin"
		})).Return(&s3.HeadObjectOutput{
			ContentLength: aws.Int64(4096),
		}, nil).Once()

		blob, err := store.Open(context.Background(), "segment.bin")
		assert.NoError(t, err)
		assert.Equal(t, int64(4096), blob.Size())
	})
}

func TestExpressStore_Put(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

	mockClient.On("PutObject", mock.Anything, mock.MatchedBy(func(input *s3.PutObjectInput) bool {
		return *input.Bucket == "express-bucket--usw2-az1--x-s3" &&
			*input.Key == "data/manifest.bin" &&
			input.IfNoneMatch == nil // Regular Put doesn't use conditional
	})).Run(func(args mock.Arguments) {
		input := args.Get(1).(*s3.PutObjectInput)
		data, _ := io.ReadAll(input.Body)
		assert.Equal(t, []byte("manifest-data"), data)
	}).Return(&s3.PutObjectOutput{}, nil).Once()

	err := store.Put(context.Background(), "manifest.bin", []byte("manifest-data"))
	assert.NoError(t, err)
}

func TestExpressStore_PutIfNotExists(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		mockClient := new(MockS3Client)
		store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

		mockClient.On("PutObject", mock.Anything, mock.MatchedBy(func(input *s3.PutObjectInput) bool {
			return *input.Bucket == "express-bucket--usw2-az1--x-s3" &&
				*input.Key == "data/new-file.bin" &&
				input.IfNoneMatch != nil && *input.IfNoneMatch == "*"
		})).Return(&s3.PutObjectOutput{}, nil).Once()

		err := store.PutIfNotExists(context.Background(), "new-file.bin", []byte("data"))
		assert.NoError(t, err)
	})

	t.Run("Conflict_PreconditionFailed", func(t *testing.T) {
		mockClient := new(MockS3Client)
		store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

		mockClient.On("PutObject", mock.Anything, mock.MatchedBy(func(input *s3.PutObjectInput) bool {
			return *input.Key == "data/existing.bin" &&
				input.IfNoneMatch != nil && *input.IfNoneMatch == "*"
		})).Return(nil, &mockAPIError{code: "PreconditionFailed"}).Once()

		err := store.PutIfNotExists(context.Background(), "existing.bin", []byte("data"))
		assert.Equal(t, ErrConflict, err)
	})

	t.Run("Conflict_ConditionalRequestConflict", func(t *testing.T) {
		mockClient := new(MockS3Client)
		store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

		mockClient.On("PutObject", mock.Anything, mock.MatchedBy(func(input *s3.PutObjectInput) bool {
			return *input.Key == "data/existing2.bin" &&
				input.IfNoneMatch != nil && *input.IfNoneMatch == "*"
		})).Return(nil, &mockAPIError{code: "ConditionalRequestConflict"}).Once()

		err := store.PutIfNotExists(context.Background(), "existing2.bin", []byte("data"))
		assert.Equal(t, ErrConflict, err)
	})

	t.Run("OtherError", func(t *testing.T) {
		mockClient := new(MockS3Client)
		store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

		mockClient.On("PutObject", mock.Anything, mock.Anything).
			Return(nil, errors.New("network error")).Once()

		err := store.PutIfNotExists(context.Background(), "file.bin", []byte("data"))
		assert.Error(t, err)
		assert.NotEqual(t, ErrConflict, err)
	})
}

// mockAPIError implements smithy.APIError for testing
type mockAPIError struct {
	code string
}

func (m *mockAPIError) Error() string                 { return m.code }
func (m *mockAPIError) ErrorCode() string             { return m.code }
func (m *mockAPIError) ErrorMessage() string          { return m.code }
func (m *mockAPIError) ErrorFault() smithy.ErrorFault { return smithy.FaultClient }

func TestExpressStore_Delete(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

	mockClient.On("DeleteObject", mock.Anything, mock.MatchedBy(func(input *s3.DeleteObjectInput) bool {
		return *input.Bucket == "express-bucket--usw2-az1--x-s3" && *input.Key == "data/old-segment"
	})).Return(&s3.DeleteObjectOutput{}, nil).Once()

	err := store.Delete(context.Background(), "old-segment")
	assert.NoError(t, err)
}

func TestExpressStore_List(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data/")

	mockClient.On("ListObjectsV2", mock.Anything, mock.MatchedBy(func(input *s3.ListObjectsV2Input) bool {
		return *input.Bucket == "express-bucket--usw2-az1--x-s3" && *input.Prefix == "data/segments"
	})).Return(&s3.ListObjectsV2Output{
		Contents: []types.Object{
			{Key: aws.String("data/segments/seg1.bin")},
			{Key: aws.String("data/segments/seg2.bin")},
		},
	}, nil).Once()

	keys, err := store.List(context.Background(), "segments")
	assert.NoError(t, err)
	assert.Equal(t, []string{"segments/seg1.bin", "segments/seg2.bin"}, keys)
}

func TestExpressStore_Create(t *testing.T) {
	mockClient := new(MockS3Client)
	store := NewExpressStore(mockClient, "express-bucket--usw2-az1--x-s3", "data")

	mockClient.On("PutObject", mock.Anything, mock.MatchedBy(func(input *s3.PutObjectInput) bool {
		return *input.Bucket == "express-bucket--usw2-az1--x-s3" && *input.Key == "data/stream.bin"
	})).Run(func(args mock.Arguments) {
		input := args.Get(1).(*s3.PutObjectInput)
		io.ReadAll(input.Body)
	}).Return(&s3.PutObjectOutput{}, nil).Once()

	wb, err := store.Create(context.Background(), "stream.bin")
	assert.NoError(t, err)

	_, err = wb.Write([]byte("streaming content"))
	assert.NoError(t, err)

	err = wb.Close()
	assert.NoError(t, err)
}
