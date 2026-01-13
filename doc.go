// Package vecgo provides a high-performance embedded vector database for Go.
//
// Vecgo is an embeddable, hybrid vector database designed for production workloads.
// It combines commit-oriented durability with HNSW + DiskANN indexing for
// best-in-class performance.
//
// # Quick Start
//
// Local mode:
//
//	db, _ := vecgo.Open(vecgo.Local("./data"), vecgo.Create(128, vecgo.MetricL2))
//	db, _ := vecgo.Open(vecgo.Local("./data"))  // re-open existing
//
// Cloud mode:
//
//	s3Store, _ := s3.New(ctx, "my-bucket", s3.WithPrefix("vectors/"))
//	db, _ := vecgo.Open(vecgo.Remote(s3Store))
//	db, _ := vecgo.Open(vecgo.Remote(s3Store), vecgo.WithCacheDir("/fast/nvme"))
//
// # Search with Data
//
// By default, search returns IDs, scores, metadata, and payload:
//
//	results, _ := db.Search(ctx, query, 10)
//	for _, r := range results {
//	    fmt.Println(r.ID, r.Score, r.Metadata, r.Payload)
//	}
//
// For minimal results (IDs + scores only), use WithoutData():
//
//	results, _ := db.Search(ctx, query, 10, vecgo.WithoutData())
//
// # Durability Model
//
// Vecgo uses commit-oriented durability (like LanceDB/Git):
//
//	db.Insert(ctx, vec, nil, nil)  // buffered in memory
//	db.Commit(ctx)                 // durable after this
//
// # Key Features
//
//   - HNSW + DiskANN hybrid indexing
//   - Commit-oriented durability (no WAL complexity)
//   - Full quantization suite (PQ, OPQ, SQ, BQ, RaBitQ, INT4)
//   - Cloud-native storage (S3/GCS/Azure via BlobStore)
//   - Time-travel queries
//   - Hybrid search (BM25 + vectors with RRF fusion)
//   - SIMD optimized (AVX-512/AVX2/NEON/SVE2)
package vecgo
