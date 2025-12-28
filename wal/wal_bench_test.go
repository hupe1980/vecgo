package wal

import (
	"testing"

	"github.com/hupe1980/vecgo/metadata"
)

// BenchmarkWALInsert benchmarks WAL insert operations with binary encoding.
func BenchmarkWALInsert(b *testing.B) {
	dir := b.TempDir()
	wal, err := New(func(o *Options) {
		o.Path = dir
		o.Compress = false
	})
	if err != nil {
		b.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()

	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i)
	}
	data := make([]byte, 100)

	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		err := wal.LogInsert(uint32(i), vector, data, nil)
		if err != nil {
			b.Fatalf("LogInsert failed: %v", err)
		}
	}
}

// BenchmarkWALInsertCompressed benchmarks WAL insert operations with compression.
func BenchmarkWALInsertCompressed(b *testing.B) {
	dir := b.TempDir()
	wal, err := New(func(o *Options) {
		o.Path = dir
		o.Compress = true
	})
	if err != nil {
		b.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()

	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i)
	}
	data := make([]byte, 100)

	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		err := wal.LogInsert(uint32(i), vector, data, nil)
		if err != nil {
			b.Fatalf("LogInsert failed: %v", err)
		}
	}
}

// BenchmarkWALBatchInsert benchmarks batch insert operations.
func BenchmarkWALBatchInsert(b *testing.B) {
	dir := b.TempDir()
	wal, err := New(func(o *Options) {
		o.Path = dir
		o.Compress = false
	})
	if err != nil {
		b.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()

	batchSize := 100
	ids := make([]uint32, batchSize)
	vectors := make([][]float32, batchSize)
	dataSlice := make([][]byte, batchSize)
	metadataSlice := make([]metadata.Metadata, batchSize)

	for i := 0; i < batchSize; i++ {
		ids[i] = uint32(i)
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = float32(j)
		}
		dataSlice[i] = make([]byte, 100)
		metadataSlice[i] = nil
	}

	b.ResetTimer()
	for b.Loop() {
		err := wal.LogBatchInsert(ids, vectors, dataSlice, metadataSlice)
		if err != nil {
			b.Fatalf("LogBatchInsert failed: %v", err)
		}
	}
}

// BenchmarkWALReplay benchmarks WAL replay operations.
func BenchmarkWALReplay(b *testing.B) {
	dir := b.TempDir()
	wal, err := New(func(o *Options) {
		o.Path = dir
		o.Compress = false
	})
	if err != nil {
		b.Fatalf("Failed to create WAL: %v", err)
	}

	// Populate with entries
	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i)
	}
	data := make([]byte, 100)

	for i := uint32(0); i < 1000; i++ {
		wal.LogInsert(i, vector, data, nil)
	}
	wal.Close()

	b.ResetTimer()
	for b.Loop() {
		wal, err := New(func(o *Options) {
			o.Path = dir
		})
		if err != nil {
			b.Fatalf("Failed to create WAL: %v", err)
		}

		count := 0
		err = wal.ReplayCommitted(func(entry Entry) error {
			count++
			return nil
		})
		if err != nil {
			b.Fatalf("ReplayCommitted failed: %v", err)
		}

		wal.Close()
	}
}
