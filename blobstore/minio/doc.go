// Package minio provides a BlobStore implementation using the MinIO client.
//
// MinIO is a high-performance, S3-compatible object storage system. This package
// uses the official MinIO Go client library for optimal compatibility with MinIO
// and other S3-compatible storage systems like Ceph, SeaweedFS, and Garage.
//
// # Basic Usage
//
//	client, err := minio.New("localhost:9000", &minio.Options{
//	    Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
//	    Secure: false,
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	store := minioblob.NewStore(client, "my-bucket", "vectors/")
//	eng, err := vecgo.Open(ctx, vecgo.Remote(store), vecgo.Create(128, vecgo.MetricL2))
//
// # Features
//
//   - Native MinIO client with optimal performance
//   - Works with any S3-compatible storage (Ceph, Garage, SeaweedFS)
//   - Streaming uploads for large segments
//   - Air-gap friendly (no AWS dependencies required)
//
// # Configuration Options
//
// The MinIO client supports various configuration options:
//
//	client, _ := minio.New("s3.example.com:9000", &minio.Options{
//	    Creds:  credentials.NewStaticV4(accessKey, secretKey, ""),
//	    Secure: true,                    // Use HTTPS
//	    Region: "us-east-1",             // Optional region
//	})
package minio
