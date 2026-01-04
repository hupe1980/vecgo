package engine

import (
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// WithFilter sets the metadata filter for the search.
func WithFilter(filter *metadata.FilterSet) func(*model.SearchOptions) {
	return func(opts *model.SearchOptions) {
		opts.Filter = filter
	}
}

// WithPreFilter forces pre-filtering (or post-filtering if false).
func WithPreFilter(preFilter bool) func(*model.SearchOptions) {
	return func(opts *model.SearchOptions) {
		opts.PreFilter = &preFilter
	}
}

// WithRefineFactor sets the refine factor for the search.
func WithRefineFactor(factor float32) func(*model.SearchOptions) {
	return func(opts *model.SearchOptions) {
		opts.RefineFactor = factor
	}
}

// WithNProbes sets the number of probes for the search.
func WithNProbes(n int) func(*model.SearchOptions) {
	return func(opts *model.SearchOptions) {
		opts.NProbes = n
	}
}

// WithVector requests the vector to be returned in the search results.
func WithVector() func(*model.SearchOptions) {
	return func(opts *model.SearchOptions) {
		opts.IncludeVector = true
	}
}

// WithMetadata requests the metadata to be returned in the search results.
func WithMetadata() func(*model.SearchOptions) {
	return func(opts *model.SearchOptions) {
		opts.IncludeMetadata = true
	}
}

// WithPayload requests the payload to be returned in the search results.
func WithPayload() func(*model.SearchOptions) {
	return func(opts *model.SearchOptions) {
		opts.IncludePayload = true
	}
}
