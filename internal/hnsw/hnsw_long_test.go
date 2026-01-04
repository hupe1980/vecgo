//go:build longtests

package hnsw

import (
	"context"
	"testing"
)

// These tests are intentionally expensive and are excluded from default test
// runs. Run with:
//
//	go test ./index/hnsw -tags=longtests -run TestValidateInsertSearchLong -count=1
func TestValidateInsertSearchLong(t *testing.T) {
	ctx := context.Background()

	tests := []TestCases{
		{
			VectorSize: 5000,
			VectorDim:  16,
			M:          16,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 5000,
			VectorDim:  32,
			M:          16,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(caseName(tc), func(t *testing.T) {
			runValidateInsertSearchCase(t, ctx, tc)
		})
	}
}
