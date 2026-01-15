// Package s3 provides S3 implementations of the blobstore.BlobStore interface.
// For AWS S3, S3 Express One Zone, and DynamoDB coordination.
//
// For MinIO and self-hosted S3-compatible storage, use the dedicated
// github.com/hupe1980/vecgo/blobstore/minio package.
//
// # Standard S3 Store
//
//	store, err := s3.New(ctx, "my-bucket",
//	    s3.WithPrefix("vectors/"),
//	    s3.WithRegion("us-east-1"),
//	)
//
//	db, err := vecgo.Open(ctx, vecgo.Remote(store), vecgo.Create(128, vecgo.MetricL2))
//
// # S3 Express One Zone (Low Latency)
//
// For Lambda, Kubernetes, or latency-sensitive workloads:
//
//	expressStore := s3.NewExpressStore(s3Client, "my-bucket--usw2-az1--x-s3", "vectors/")
//	db, err := vecgo.Open(ctx, vecgo.Remote(expressStore))
//
// S3 Express provides single-digit millisecond latency and supports conditional writes.
//
// # DynamoDB Commit Store (Concurrent Writers)
//
// S3 lacks atomic writes. For safe concurrent writers:
//
//	s3Store := s3.NewStore(s3Client, "my-bucket", "vectors/")
//	commitStore := s3.NewDDBCommitStore(s3Store, ddbClient, "vecgo-commits", "s3://my-bucket/vectors/")
//	db, err := vecgo.Open(ctx, vecgo.Remote(commitStore))
//
// The DynamoDB table must have partition key "base_uri" (string) and sort key "version" (number).
//
// # Features
//
//   - Range reads for efficient partial fetches
//   - Multipart uploads for large segments
//   - Automatic pagination for listing
//   - Configurable prefix for multi-tenant isolation
//   - S3 Express One Zone for low-latency access
//   - DynamoDB coordination for concurrent writers
package s3
