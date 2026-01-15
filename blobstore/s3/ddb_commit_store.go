package s3

import (
	"bytes"
	"context"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/hupe1980/vecgo/blobstore"
)

// DDBCommitStore implements blobstore.BlobStore backed by S3 with DynamoDB
// for atomic manifest commits. This enables safe concurrent writers.
//
// DynamoDB is used as a commit log for manifest updates, providing the
// atomic compare-and-swap semantics that S3 lacks. The commit store:
//   - Writes manifest content to S3
//   - Uses DynamoDB conditional writes to atomically update the "CURRENT" pointer
//   - Enables multiple writers to safely coordinate without data loss
//
// Table schema:
//   - Partition key: base_uri (string) - the S3 prefix/path
//   - Sort key: version (number) - monotonically increasing version
//
// Create table with:
//
//	aws dynamodb create-table \
//	  --table-name vecgo-commits \
//	  --attribute-definitions AttributeName=base_uri,AttributeType=S AttributeName=version,AttributeType=N \
//	  --key-schema AttributeName=base_uri,KeyType=HASH AttributeName=version,KeyType=RANGE \
//	  --billing-mode PAY_PER_REQUEST
type DDBCommitStore struct {
	s3Store   *Store
	ddbClient DDBClient
	tableName string
	baseURI   string // S3 bucket/prefix used as partition key
}

// DDBClient is the interface for DynamoDB operations.
type DDBClient interface {
	PutItem(ctx context.Context, params *dynamodb.PutItemInput, optFns ...func(*dynamodb.Options)) (*dynamodb.PutItemOutput, error)
	Query(ctx context.Context, params *dynamodb.QueryInput, optFns ...func(*dynamodb.Options)) (*dynamodb.QueryOutput, error)
	GetItem(ctx context.Context, params *dynamodb.GetItemInput, optFns ...func(*dynamodb.Options)) (*dynamodb.GetItemOutput, error)
	DeleteItem(ctx context.Context, params *dynamodb.DeleteItemInput, optFns ...func(*dynamodb.Options)) (*dynamodb.DeleteItemOutput, error)
}

// ErrConcurrentModification is returned when a concurrent write is detected.
var ErrConcurrentModification = errors.New("concurrent modification detected")

// NewDDBCommitStore creates a new S3+DynamoDB commit store.
// The baseURI should be "s3://bucket/prefix" format used as partition key.
func NewDDBCommitStore(s3Store *Store, ddbClient DDBClient, tableName, baseURI string) *DDBCommitStore {
	return &DDBCommitStore{
		s3Store:   s3Store,
		ddbClient: ddbClient,
		tableName: tableName,
		baseURI:   baseURI,
	}
}

// Open opens a blob for reading.
func (s *DDBCommitStore) Open(ctx context.Context, name string) (blobstore.Blob, error) {
	// For CURRENT file, read from DynamoDB to get the latest version
	if name == "CURRENT" {
		version, manifestPath, err := s.getLatestVersion(ctx)
		if err != nil {
			return nil, err
		}
		if version == 0 {
			return nil, blobstore.ErrNotFound
		}
		// Return a virtual blob that reads the manifest path
		return &virtualCurrentBlob{content: []byte(manifestPath)}, nil
	}
	return s.s3Store.Open(ctx, name)
}

// Put writes a blob. For CURRENT, uses DynamoDB conditional write.
func (s *DDBCommitStore) Put(ctx context.Context, name string, data []byte) error {
	if name == "CURRENT" {
		return s.commitVersion(ctx, string(data))
	}
	return s.s3Store.Put(ctx, name, data)
}

// Create creates a writable blob.
func (s *DDBCommitStore) Create(ctx context.Context, name string) (blobstore.WritableBlob, error) {
	return s.s3Store.Create(ctx, name)
}

// Delete deletes a blob.
func (s *DDBCommitStore) Delete(ctx context.Context, name string) error {
	return s.s3Store.Delete(ctx, name)
}

// List lists blobs with prefix.
func (s *DDBCommitStore) List(ctx context.Context, prefix string) ([]string, error) {
	return s.s3Store.List(ctx, prefix)
}

// getLatestVersion queries DynamoDB for the latest committed version.
func (s *DDBCommitStore) getLatestVersion(ctx context.Context) (uint64, string, error) {
	resp, err := s.ddbClient.Query(ctx, &dynamodb.QueryInput{
		TableName:              aws.String(s.tableName),
		KeyConditionExpression: aws.String("base_uri = :uri"),
		ExpressionAttributeValues: map[string]types.AttributeValue{
			":uri": &types.AttributeValueMemberS{Value: s.baseURI},
		},
		ScanIndexForward: aws.Bool(false), // Descending order
		Limit:            aws.Int32(1),
	})
	if err != nil {
		return 0, "", fmt.Errorf("failed to query DynamoDB: %w", err)
	}

	if len(resp.Items) == 0 {
		return 0, "", nil
	}

	item := resp.Items[0]
	versionAttr, ok := item["version"].(*types.AttributeValueMemberN)
	if !ok {
		return 0, "", errors.New("invalid version attribute in DynamoDB")
	}
	pathAttr, ok := item["manifest_path"].(*types.AttributeValueMemberS)
	if !ok {
		return 0, "", errors.New("invalid manifest_path attribute in DynamoDB")
	}

	var version uint64
	if _, err := fmt.Sscanf(versionAttr.Value, "%d", &version); err != nil {
		return 0, "", fmt.Errorf("failed to parse version: %w", err)
	}

	return version, pathAttr.Value, nil
}

// commitVersion atomically commits a new manifest version using DynamoDB conditional write.
func (s *DDBCommitStore) commitVersion(ctx context.Context, manifestPath string) error {
	// Get current version
	currentVersion, _, err := s.getLatestVersion(ctx)
	if err != nil {
		return err
	}

	newVersion := currentVersion + 1

	// Conditional put: only succeed if this version doesn't exist yet
	_, err = s.ddbClient.PutItem(ctx, &dynamodb.PutItemInput{
		TableName: aws.String(s.tableName),
		Item: map[string]types.AttributeValue{
			"base_uri":      &types.AttributeValueMemberS{Value: s.baseURI},
			"version":       &types.AttributeValueMemberN{Value: fmt.Sprintf("%d", newVersion)},
			"manifest_path": &types.AttributeValueMemberS{Value: manifestPath},
		},
		ConditionExpression: aws.String("attribute_not_exists(version)"),
	})

	if err != nil {
		var condErr *types.ConditionalCheckFailedException
		if errors.As(err, &condErr) {
			return ErrConcurrentModification
		}
		return fmt.Errorf("failed to commit version to DynamoDB: %w", err)
	}

	return nil
}

// virtualCurrentBlob is a simple in-memory blob for the CURRENT file content.
type virtualCurrentBlob struct {
	content []byte
}

func (b *virtualCurrentBlob) Close() error {
	return nil
}

func (b *virtualCurrentBlob) Size() int64 {
	return int64(len(b.content))
}

func (b *virtualCurrentBlob) ReadAt(ctx context.Context, p []byte, off int64) (int, error) {
	if off >= int64(len(b.content)) {
		return 0, nil
	}
	n := copy(p, b.content[off:])
	return n, nil
}

func (b *virtualCurrentBlob) ReadRange(ctx context.Context, off, lenReq int64) (blobstore.ReadCloser, error) {
	if off >= int64(len(b.content)) {
		return blobstore.NopReadCloser(bytes.NewReader(nil)), nil
	}
	end := off + lenReq
	if end > int64(len(b.content)) {
		end = int64(len(b.content))
	}
	return blobstore.NopReadCloser(bytes.NewReader(b.content[off:end])), nil
}
