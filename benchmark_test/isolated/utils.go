package isolated

import (
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/testutil"
)

// toTestUtilResults converts index.SearchResult to testutil.SearchResult
func toTestUtilResults(results []index.SearchResult) []testutil.SearchResult {
	out := make([]testutil.SearchResult, len(results))
	for i, r := range results {
		out[i] = testutil.SearchResult{ID: uint64(r.ID), Distance: r.Distance}
	}
	return out
}
