package benchmark_test

import (
	"context"
	"fmt"
	"sort"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/lexical"
	"github.com/hupe1980/vecgo/lexical/bm25"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkHybridSearch(b *testing.B) {
	dim := 128
	numVecs := 10000
	vocab := []string{"apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"}

	dir := b.TempDir()
	lexIdx := bm25.New()
	e, _ := vecgo.Open(vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2), vecgo.WithLexicalIndex(lexIdx, "text"))
	defer e.Close()

	rng := testutil.NewRNG(1)
	data := make([][]float32, numVecs)
	pks := make([]model.ID, numVecs)
	for i := 0; i < numVecs; i++ {
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		data[i] = vec
		// Pick 2 random words
		w1 := vocab[rng.Intn(len(vocab))]
		w2 := vocab[rng.Intn(len(vocab))]
		text := fmt.Sprintf("%s %s", w1, w2)

		id, _ := e.Insert(vec, metadata.Document{"text": metadata.String(text)}, nil)
		pks[i] = id
	}

	ctx := context.Background()
	qVec := make([]float32, dim)
	rng.FillUniform(qVec)
	qText := "apple banana"

	b.ResetTimer()

	b.Run("VectorOnly", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			e.Search(ctx, qVec, 10)
		}
		b.StopTimer()
		truth := exactTopK_L2_WithIDs(data, pks, qVec, 10)
		res, _ := e.Search(ctx, qVec, 10)
		b.ReportMetric(recallAtK(res, truth), "recall@10")
	})

	b.Run("Hybrid", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			e.HybridSearch(ctx, qVec, qText, 10, 60)
		}
		b.StopTimer()
		truth := naiveHybridSearch(data, pks, lexIdx, qVec, qText, 10, 60)
		res, _ := e.HybridSearch(ctx, qVec, qText, 10, 60)
		b.ReportMetric(recallAtK(res, truth), "recall@10")
	})

	b.Run("DAAT", func(b *testing.B) {
		// Truth (TAAT)
		truthCands, _ := lexIdx.Search(qText, 10)
		truth := make([]model.ID, len(truthCands))
		for i, c := range truthCands {
			truth[i] = c.ID
		}

		for i := 0; i < b.N; i++ {
			lexIdx.SearchDAAT(qText, 10)
		}
		b.StopTimer()

		res, _ := lexIdx.SearchDAAT(qText, 10)
		b.ReportMetric(recallAtK(res, truth), "recall@10")
	})
}

func naiveHybridSearch(data [][]float32, pks []model.ID, lexIdx lexical.Index, qVec []float32, qText string, k int, rrfK int) []model.ID {
	// 1. Exact Vector Search
	vectorK := k * 2
	if vectorK < 50 {
		vectorK = 50
	}
	// exactTopK_L2 returns PKs sorted by distance (best first)
	vecPKs := exactTopK_L2_WithIDs(data, pks, qVec, vectorK)

	// 2. Lexical Search
	lexResults, _ := lexIdx.Search(qText, vectorK)

	// 3. RRF Fusion
	finalScores := make(map[model.ID]float32)

	// Vector Ranks
	for rank, pk := range vecPKs {
		score := 1.0 / float32(rrfK+rank+1)
		finalScores[pk] = score
	}

	// Lexical Ranks
	for rank, c := range lexResults {
		score := 1.0 / float32(rrfK+rank+1)
		finalScores[c.ID] += score
	}

	// 4. Sort Final
	type candidate struct {
		pk    model.ID
		score float32
	}
	candidates := make([]candidate, 0, len(finalScores))
	for pk, score := range finalScores {
		candidates = append(candidates, candidate{pk: pk, score: score})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	// Top K
	out := make([]model.ID, 0, k)
	for i := 0; i < k && i < len(candidates); i++ {
		out = append(out, candidates[i].pk)
	}
	return out
}
