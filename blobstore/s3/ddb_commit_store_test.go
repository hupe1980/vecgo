package s3

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/hupe1980/vecgo/blobstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockDDBClient is an in-memory DynamoDB mock for testing.
type mockDDBClient struct {
	mu    sync.RWMutex
	items map[string]map[string]types.AttributeValue // key -> item
}

func newMockDDBClient() *mockDDBClient {
	return &mockDDBClient{
		items: make(map[string]map[string]types.AttributeValue),
	}
}

func (m *mockDDBClient) PutItem(ctx context.Context, params *dynamodb.PutItemInput, optFns ...func(*dynamodb.Options)) (*dynamodb.PutItemOutput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	baseURI := params.Item["base_uri"].(*types.AttributeValueMemberS).Value
	version := params.Item["version"].(*types.AttributeValueMemberN).Value
	key := baseURI + ":" + version

	// Check conditional expression
	if params.ConditionExpression != nil && *params.ConditionExpression == "attribute_not_exists(version)" {
		if _, exists := m.items[key]; exists {
			return nil, &types.ConditionalCheckFailedException{Message: aws.String("condition failed")}
		}
	}

	m.items[key] = params.Item
	return &dynamodb.PutItemOutput{}, nil
}

func (m *mockDDBClient) Query(ctx context.Context, params *dynamodb.QueryInput, optFns ...func(*dynamodb.Options)) (*dynamodb.QueryOutput, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	baseURI := params.ExpressionAttributeValues[":uri"].(*types.AttributeValueMemberS).Value

	// Find items matching baseURI, sort by version descending
	var items []map[string]types.AttributeValue
	for _, item := range m.items {
		if item["base_uri"].(*types.AttributeValueMemberS).Value == baseURI {
			items = append(items, item)
		}
	}

	// Sort descending by version
	for i := 0; i < len(items)-1; i++ {
		for j := i + 1; j < len(items); j++ {
			vi := items[i]["version"].(*types.AttributeValueMemberN).Value
			vj := items[j]["version"].(*types.AttributeValueMemberN).Value
			if vi < vj {
				items[i], items[j] = items[j], items[i]
			}
		}
	}

	if params.Limit != nil && int(*params.Limit) < len(items) {
		items = items[:*params.Limit]
	}

	return &dynamodb.QueryOutput{Items: items}, nil
}

func (m *mockDDBClient) GetItem(ctx context.Context, params *dynamodb.GetItemInput, optFns ...func(*dynamodb.Options)) (*dynamodb.GetItemOutput, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	baseURI := params.Key["base_uri"].(*types.AttributeValueMemberS).Value
	version := params.Key["version"].(*types.AttributeValueMemberN).Value
	key := baseURI + ":" + version

	if item, ok := m.items[key]; ok {
		return &dynamodb.GetItemOutput{Item: item}, nil
	}
	return &dynamodb.GetItemOutput{}, nil
}

func (m *mockDDBClient) DeleteItem(ctx context.Context, params *dynamodb.DeleteItemInput, optFns ...func(*dynamodb.Options)) (*dynamodb.DeleteItemOutput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	baseURI := params.Key["base_uri"].(*types.AttributeValueMemberS).Value
	version := params.Key["version"].(*types.AttributeValueMemberN).Value
	delete(m.items, baseURI+":"+version)
	return &dynamodb.DeleteItemOutput{}, nil
}

func newTestDDBCommitStore(ddb *mockDDBClient, baseURI string) *DDBCommitStore {
	s3Store := &Store{
		client: &MockS3Client{},
		bucket: "test-bucket",
		prefix: "test/",
	}
	return NewDDBCommitStore(s3Store, ddb, "vecgo-commits", baseURI)
}

func TestDDBCommitStore_FirstCommit(t *testing.T) {
	ctx := context.Background()
	ddb := newMockDDBClient()
	store := newTestDDBCommitStore(ddb, "s3://test-bucket/test/")

	// First commit should succeed
	err := store.Put(ctx, "CURRENT", []byte("MANIFEST-00001.bin"))
	require.NoError(t, err)

	// Read back CURRENT
	blob, err := store.Open(ctx, "CURRENT")
	require.NoError(t, err)
	defer blob.Close()

	buf := make([]byte, 100)
	n, _ := blob.ReadAt(ctx, buf, 0)
	assert.Equal(t, "MANIFEST-00001.bin", string(buf[:n]))
}

func TestDDBCommitStore_MultipleCommits(t *testing.T) {
	ctx := context.Background()
	ddb := newMockDDBClient()
	store := newTestDDBCommitStore(ddb, "s3://test-bucket/test/")

	// Commit versions 1, 2, 3
	for i := 1; i <= 3; i++ {
		err := store.Put(ctx, "CURRENT", []byte(fmt.Sprintf("MANIFEST-%05d.bin", i)))
		require.NoError(t, err)
	}

	// Read back should get latest (version 3)
	blob, err := store.Open(ctx, "CURRENT")
	require.NoError(t, err)
	defer blob.Close()

	buf := make([]byte, 100)
	n, _ := blob.ReadAt(ctx, buf, 0)
	assert.Equal(t, "MANIFEST-00003.bin", string(buf[:n]))
}

func TestDDBCommitStore_ConcurrentCommits(t *testing.T) {
	ctx := context.Background()
	ddb := newMockDDBClient()
	store := newTestDDBCommitStore(ddb, "s3://test-bucket/test/")

	// Initial commit
	err := store.Put(ctx, "CURRENT", []byte("MANIFEST-00001.bin"))
	require.NoError(t, err)

	// Concurrent writers
	var wg sync.WaitGroup
	successes := 0
	conflicts := 0
	var mu sync.Mutex

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			err := store.Put(ctx, "CURRENT", []byte(fmt.Sprintf("MANIFEST-%05d.bin", id+2)))
			mu.Lock()
			defer mu.Unlock()
			if err == ErrConcurrentModification {
				conflicts++
			} else if err == nil {
				successes++
			} else {
				t.Errorf("unexpected error: %v", err)
			}
		}(i)
	}

	wg.Wait()
	assert.Greater(t, successes, 0, "at least one writer should succeed")
	t.Logf("successes: %d, conflicts: %d", successes, conflicts)
}

func TestDDBCommitStore_NotFoundBeforeCommit(t *testing.T) {
	ctx := context.Background()
	ddb := newMockDDBClient()
	store := newTestDDBCommitStore(ddb, "s3://test-bucket/test/")

	_, err := store.Open(ctx, "CURRENT")
	require.ErrorIs(t, err, blobstore.ErrNotFound)
}

func TestDDBCommitStore_IsolatedNamespaces(t *testing.T) {
	ctx := context.Background()
	ddb := newMockDDBClient()

	store1 := newTestDDBCommitStore(ddb, "s3://bucket-a/path/")
	store2 := newTestDDBCommitStore(ddb, "s3://bucket-b/path/")

	// Commit to each store
	require.NoError(t, store1.Put(ctx, "CURRENT", []byte("MANIFEST-A.bin")))
	require.NoError(t, store2.Put(ctx, "CURRENT", []byte("MANIFEST-B.bin")))

	// Each sees their own manifest
	blob1, _ := store1.Open(ctx, "CURRENT")
	buf := make([]byte, 100)
	n, _ := blob1.ReadAt(ctx, buf, 0)
	assert.Equal(t, "MANIFEST-A.bin", string(buf[:n]))
	blob1.Close()

	blob2, _ := store2.Open(ctx, "CURRENT")
	n, _ = blob2.ReadAt(ctx, buf, 0)
	assert.Equal(t, "MANIFEST-B.bin", string(buf[:n]))
	blob2.Close()
}
