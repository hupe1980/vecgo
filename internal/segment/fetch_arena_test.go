package segment_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// mockSegment implements segment.Segment for benchmarking FetchArena
type mockSegment struct {
	dim int
}

func (m *mockSegment) ID() model.SegmentID { return 1 }
func (m *mockSegment) RowCount() uint32    { return 10000 }
func (m *mockSegment) Metric() any         { return nil }
func (m *mockSegment) HasGraphIndex() bool { return false }
func (m *mockSegment) Search(ctx context.Context, q []float32, k int, filter any, opts model.SearchOptions, s any) error {
	return nil
}
func (m *mockSegment) GetID(ctx context.Context, rowID uint32) (model.ID, bool) {
	return model.ID(rowID), true
}
func (m *mockSegment) Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error) {
	return dst, nil
}
func (m *mockSegment) FetchIDs(ctx context.Context, rows []uint32, dst []model.ID) error {
	for i, r := range rows {
		dst[i] = model.ID(r)
	}
	return nil
}
func (m *mockSegment) FetchVectorsInto(ctx context.Context, rows []uint32, dim int, dst []float32) ([]bool, error) {
	return nil, nil
}
func (m *mockSegment) FetchVectorDirect(rowID uint32) []float32 { return nil }
func (m *mockSegment) Iterate(ctx context.Context, fn func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error) error {
	return nil
}
func (m *mockSegment) EvaluateFilter(ctx context.Context, filter any) (any, error) {
	return nil, nil
}
func (m *mockSegment) Advise(pattern segment.AccessPattern) error { return nil }
func (m *mockSegment) Size() int64                                { return 0 }
func (m *mockSegment) Close() error                               { return nil }

// BenchmarkFetchArena_Reuse benchmarks the FetchArena with pooling
func BenchmarkFetchArena_Reuse(b *testing.B) {
	batchSizes := []int{10, 50, 100}

	for _, batchSize := range batchSizes {
		b.Run(testName("BatchSize", batchSize), func(b *testing.B) {
			dim := 128

			// Pre-warm the pool
			arena := segment.NewFetchArenaWithCapacity(batchSize, dim)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Simulate batch fetch
				arena.EnsureCapacity(batchSize, dim)
				arena.Reset(batchSize)
				arena.SetDimension(dim)

				// Allocate slices
				arena.IDs = arena.IDs[:batchSize]
				arena.Vectors = arena.Vectors[:batchSize]
				arena.Metadatas = arena.Metadatas[:batchSize]

				for j := 0; j < batchSize; j++ {
					arena.IDs[j] = model.ID(j)
					arena.Vectors[j] = arena.AllocVectorSlice(j)
					md := arena.AcquireMetadata(j)
					(*md)["key"] = metadata.Int(int64(j))
					arena.Metadatas[j] = *md
				}

				// Build batch (simulates return to caller, but batch is only valid until Reset)
				_ = arena.BuildRecordBatch(true, true, false)
			}
		})
	}
}

// BenchmarkFetchArena_NoReuse benchmarks allocating fresh slices each time (baseline)
func BenchmarkFetchArena_NoReuse(b *testing.B) {
	batchSizes := []int{10, 50, 100}

	for _, batchSize := range batchSizes {
		b.Run(testName("BatchSize", batchSize), func(b *testing.B) {
			dim := 128

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Simulate batch fetch without arena (current approach)
				ids := make([]model.ID, batchSize)
				vectors := make([][]float32, batchSize)
				metadatas := make([]metadata.Document, batchSize)
				vectorBacking := make([]float32, batchSize*dim)

				for j := 0; j < batchSize; j++ {
					ids[j] = model.ID(j)
					vectors[j] = vectorBacking[j*dim : (j+1)*dim]
					md := make(metadata.Document, 1)
					md["key"] = metadata.Int(int64(j))
					metadatas[j] = md
				}

				// Build batch
				_ = &segment.SimpleRecordBatch{
					IDs:       ids,
					Vectors:   vectors,
					Metadatas: metadatas,
				}
			}
		})
	}
}

// BenchmarkFetchArenaPool benchmarks Get/Put from pool
func BenchmarkFetchArenaPool(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		arena := segment.GetFetchArena()
		arena.EnsureCapacity(100, 128)
		arena.Reset(100)
		segment.PutFetchArena(arena)
	}
}

func testName(prefix string, value int) string {
	return prefix + "=" + itoa(value)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	s := ""
	for i > 0 {
		s = string(rune('0'+i%10)) + s
		i /= 10
	}
	return s
}
