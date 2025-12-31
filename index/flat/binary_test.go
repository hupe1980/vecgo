package flat

import (
	"bytes"
	"context"
	"os"
	"testing"

	"github.com/hupe1980/vecgo/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBinaryPersistence_SaveLoad(t *testing.T) {
	// Create a flat index and add vectors
	f, err := New(func(o *Options) {
		o.Dimension = 128
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)

	ctx := context.Background()

	// Insert 100 vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i*128 + j)
		}
		_, err = f.Insert(ctx, vec)
		require.NoError(t, err)
	}

	// Save to file
	tmpfile, err := os.CreateTemp("", "flat-test-*.bin")
	require.NoError(t, err)
	defer os.Remove(tmpfile.Name())
	tmpfile.Close()

	err = f.SaveToFile(tmpfile.Name())
	require.NoError(t, err)

	// Load from file
	loaded, err := LoadFromFile(tmpfile.Name(), Options{Dimension: 128, DistanceType: index.DistanceTypeSquaredL2})
	require.NoError(t, err)

	// Count nodes in both indexes
	origCount := 0
	maxID := f.maxID.Load()
	for i := uint64(0); i < maxID; i++ {
		if !f.deleted.Test(i) {
			origCount++
		}
	}

	loadedCount := 0
	loadedMaxID := loaded.maxID.Load()
	for i := uint64(0); i < loadedMaxID; i++ {
		if !loaded.deleted.Test(i) {
			loadedCount++
		}
	}
	assert.Equal(t, origCount, loadedCount)

	// Test search on loaded index
	query := make([]float32, 128)
	for j := range query {
		query[j] = float32(50*128 + j) // Should match vector 50
	}

	results, err := loaded.KNNSearch(ctx, query, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(50), results[0].ID)
}

func TestBinaryPersistence_MarshalUnmarshal(t *testing.T) {
	// Create a flat index and add vectors
	f, err := New(func(o *Options) {
		o.Dimension = 64
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)

	ctx := context.Background()

	// Insert 50 vectors
	for i := 0; i < 50; i++ {
		vec := make([]float32, 64)
		for j := range vec {
			vec[j] = float32(i*64 + j)
		}
		_, err = f.Insert(ctx, vec)
		require.NoError(t, err)
	}

	// Marshal to bytes
	var buf bytes.Buffer
	_, err = f.WriteTo(&buf)
	require.NoError(t, err)

	// Unmarshal from bytes
	loaded, err := New(func(o *Options) {
		o.Dimension = 64
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)
	err = loaded.ReadFromWithOptions(&buf, Options{Dimension: 64, DistanceType: index.DistanceTypeSquaredL2})
	require.NoError(t, err)

	// Count nodes
	origCount := 0
	maxID := f.maxID.Load()
	for i := uint64(0); i < maxID; i++ {
		if !f.deleted.Test(i) {
			origCount++
		}
	}

	loadedCount := 0
	loadedMaxID := loaded.maxID.Load()
	for i := uint64(0); i < loadedMaxID; i++ {
		if !loaded.deleted.Test(i) {
			loadedCount++
		}
	}
	assert.Equal(t, origCount, loadedCount)

	// Test search on loaded index
	query := make([]float32, 64)
	for j := range query {
		query[j] = float32(25*64 + j) // Should match vector 25
	}

	results, err := loaded.KNNSearch(ctx, query, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(25), results[0].ID)
}

func TestBinaryPersistence_WithDeletes(t *testing.T) {
	// Create a flat index
	f, err := New(func(o *Options) {
		o.Dimension = 32
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)

	ctx := context.Background()

	// Insert 20 vectors
	ids := make([]uint64, 20)
	for i := 0; i < 20; i++ {
		vec := make([]float32, 32)
		for j := range vec {
			vec[j] = float32(i*32 + j)
		}
		id, err := f.Insert(ctx, vec)
		require.NoError(t, err)
		ids[i] = id
	}

	// Delete some vectors
	for i := 5; i < 15; i++ {
		err := f.Delete(ctx, ids[i])
		require.NoError(t, err)
	}

	// Save and load
	var buf bytes.Buffer
	_, err = f.WriteTo(&buf)
	require.NoError(t, err)

	loaded, err := New(func(o *Options) {
		o.Dimension = 32
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)
	err = loaded.ReadFromWithOptions(&buf, Options{Dimension: 32, DistanceType: index.DistanceTypeSquaredL2})
	require.NoError(t, err)

	// Count active nodes in loaded index (should have 10 vectors: 0-4, 15-19)
	loadedCount := 0
	loadedMaxID := loaded.maxID.Load()
	for i := uint64(0); i < loadedMaxID; i++ {
		if !loaded.deleted.Test(i) {
			loadedCount++
		}
	}
	assert.Equal(t, 10, loadedCount)

	// Search for existing vector
	query := make([]float32, 32)
	for j := range query {
		query[j] = float32(2*32 + j) // Vector 2 should exist
	}

	results, err := loaded.KNNSearch(ctx, query, 1, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(2), results[0].ID)
}

func TestBinaryPersistence_LoadUsesPersistedDistanceType(t *testing.T) {
	// Persist Cosine in the binary, then load with a different caller option.
	f, err := New(func(o *Options) {
		o.Dimension = 8
		o.DistanceType = index.DistanceTypeCosine
	})
	require.NoError(t, err)

	ctx := context.Background()
	for i := 0; i < 10; i++ {
		vec := make([]float32, 8)
		for j := range vec {
			vec[j] = float32(i*8 + j)
		}
		_, err = f.Insert(ctx, vec)
		require.NoError(t, err)
	}

	var buf bytes.Buffer
	_, err = f.WriteTo(&buf)
	require.NoError(t, err)

	loaded, err := New(func(o *Options) {
		o.Dimension = 8
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	require.NoError(t, err)
	err = loaded.ReadFromWithOptions(&buf, Options{Dimension: 8, DistanceType: index.DistanceTypeSquaredL2})
	require.NoError(t, err)
	assert.Equal(t, index.DistanceTypeCosine, loaded.opts.DistanceType)
}

func BenchmarkBinarySave(b *testing.B) {
	f, err := New(func(o *Options) {
		o.Dimension = 128
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}

	// Insert 1000 vectors
	ctx := context.Background()
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i*128 + j)
		}
		f.Insert(ctx, vec)
	}

	b.ResetTimer()
	for b.Loop() {
		var buf bytes.Buffer
		if _, err := f.WriteTo(&buf); err != nil {
			b.Fatalf("WriteTo failed: %v", err)
		}
	}
}

func BenchmarkBinaryLoad(b *testing.B) {
	f, err := New(func(o *Options) {
		o.Dimension = 128
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		b.Fatal(err)
	}

	// Insert 1000 vectors
	ctx := context.Background()
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i*128 + j)
		}
		f.Insert(ctx, vec)
	}

	// Serialize once
	var buf bytes.Buffer
	_, _ = f.WriteTo(&buf)
	data := buf.Bytes()

	b.ResetTimer()
	for b.Loop() {
		loaded, err := New(func(o *Options) {
			o.Dimension = 128
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			b.Fatal(err)
		}
		reader := bytes.NewReader(data)
		if err := loaded.ReadFromWithOptions(reader, Options{Dimension: 128, DistanceType: index.DistanceTypeSquaredL2}); err != nil {
			b.Fatalf("ReadFrom failed: %v", err)
		}
	}
}
