// Package s3 provides an S3 implementation of the blobstore.BlobStore interface.
//
// # Usage
//
//	store, err := s3.New(ctx, "my-bucket",
//	    s3.WithPrefix("vectors/"),
//	    s3.WithRegion("us-east-1"),
//	)
//
//	db, err := vecgo.Open(vecgo.Remote(store), vecgo.Create(128, vecgo.MetricL2))
//
// # Features
//
//   - Range reads for efficient partial fetches
//   - Multipart uploads for large segments
//   - Automatic pagination for listing
//   - Configurable prefix for multi-tenant isolation
package s3
