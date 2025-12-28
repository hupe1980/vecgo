package vecgo_bench_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/wal"
)

// BenchmarkSnapshotSave benchmarks saving database to disk
func BenchmarkSnapshotSave(b *testing.B) {
	sizes := []int{1000, 10000}
	dim := 384

	for _, size := range sizes {
		b.Run(formatCount(size), func(b *testing.B) {
			db := setupFlatIndex(b, dim, size)
			defer db.Close()

			tmpDir := b.TempDir()
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				snapshotPath := filepath.Join(tmpDir, "snapshot.bin")
				err := db.SaveToFile(snapshotPath)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkSnapshotLoad benchmarks snapshot loading using zero-copy mmap.
// Note: Regular (non-mmap) loading was removed due to 153GB allocation for 10K vectors.
func BenchmarkSnapshotLoad(b *testing.B) {
	sizes := []int{1000, 10000}
	dim := 384

	for _, size := range sizes {
		b.Run(formatCount(size), func(b *testing.B) {
			tmpDir := b.TempDir()
			snapshotPath := filepath.Join(tmpDir, "snapshot.bin")

			// Create snapshot
			db := setupFlatIndex(b, dim, size)
			err := db.SaveToFile(snapshotPath)
			if err != nil {
				b.Fatal(err)
			}
			db.Close()

			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				db, err := vecgo.NewFromFile[int](snapshotPath)
				if err != nil {
					b.Fatal(err)
				}
				db.Close()
			}
		})
	}
}

// BenchmarkWALWrite benchmarks WAL write performance
func BenchmarkWALWrite(b *testing.B) {
	modes := []struct {
		name string
		mode wal.DurabilityMode
	}{
		{"Async", wal.DurabilityAsync},
		{"GroupCommit", wal.DurabilityGroupCommit},
		{"Sync", wal.DurabilitySync},
	}
	dim := 384

	for _, m := range modes {
		b.Run(m.name, func(b *testing.B) {
			tmpDir := b.TempDir()

			db, err := vecgo.Flat[int](dim).
				SquaredL2().
				WAL(tmpDir, func(o *wal.Options) {
					o.DurabilityMode = m.mode
					o.GroupCommitInterval = 10 // 10ms for group commit
					o.GroupCommitMaxOps = 100
				}).
				Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				vec := randomVector(dim)
				_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
					Vector: vec,
					Data:   i,
				})
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkWALReplay benchmarks WAL recovery
func BenchmarkWALReplay(b *testing.B) {
	sizes := []int{100, 1000, 10000}
	dim := 384

	for _, size := range sizes {
		b.Run(formatCount(size), func(b *testing.B) {
			tmpDir := b.TempDir()
			snapshotPath := filepath.Join(tmpDir, "snapshot.bin")
			walPath := filepath.Join(tmpDir, "wal")

			// Setup
			db, err := vecgo.Flat[int](dim).
				SquaredL2().
				WAL(walPath, func(o *wal.Options) {
					o.DurabilityMode = wal.DurabilityAsync
				}).
				Build()
			if err != nil {
				b.Fatal(err)
			}

			ctx := b.Context()
			for i := 0; i < size; i++ {
				_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
					Vector: randomVector(dim),
					Data:   i,
				})
				if err != nil {
					b.Fatal(err)
				}
			}

			db.SaveToFile(snapshotPath)
			db.Close()

			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				db, err := vecgo.NewFromFile[int](snapshotPath, vecgo.WithWAL(walPath))
				if err != nil {
					b.Fatal(err)
				}

				err = db.RecoverFromWAL(ctx)
				if err != nil {
					b.Fatal(err)
				}

				db.Close()
			}
		})
	}
}
func BenchmarkWALCompression(b *testing.B) {
	dim := 384
	tmpDir := b.TempDir()

	b.Run("Uncompressed", func(b *testing.B) {
		db, err := vecgo.Flat[int](dim).
			SquaredL2().
			WAL(filepath.Join(tmpDir, "uncompressed"), func(o *wal.Options) {
				o.DurabilityMode = wal.DurabilityAsync
				o.Compress = false
			}).
			Build()
		if err != nil {
			b.Fatal(err)
		}
		defer db.Close()

		ctx := b.Context()
		b.ResetTimer()

		for i := 0; b.Loop(); i++ {
			vec := randomVector(dim)
			_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
				Vector: vec,
				Data:   i,
			})
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Compressed", func(b *testing.B) {
		db, err := vecgo.Flat[int](dim).
			SquaredL2().
			WAL(filepath.Join(tmpDir, "compressed"), func(o *wal.Options) {
				o.DurabilityMode = wal.DurabilityAsync
				o.Compress = true
			}).
			Build()
		if err != nil {
			b.Fatal(err)
		}
		defer db.Close()

		ctx := b.Context()
		b.ResetTimer()

		for i := 0; b.Loop(); i++ {
			vec := randomVector(dim)
			_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
				Vector: vec,
				Data:   i,
			})
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
