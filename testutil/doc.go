// Package testutil provides testing utilities for Vecgo.
//
// This package is intended for use in tests and benchmarks only.
// It provides helpers for generating random vectors, computing exact
// nearest neighbors, and verifying search recall.
//
// # Random Vector Generation
//
//	rng := testutil.NewRNG(seed)
//	vec := make([]float32, 128)
//	rng.FillUniform(vec)      // uniform [0, 1)
//	rng.FillGaussian(vec)     // standard normal
//
// # Exact Search (Ground Truth)
//
//	results := testutil.ExactTopK(query, dataset, k, distance.SquaredL2)
//
// # Recall Verification
//
//	recall := testutil.ComputeRecall(approxResults, exactResults)
package testutil
