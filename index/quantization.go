package index

// ProductQuantizationConfig configures Product Quantization (PQ) for an index.
//
// PQ is intentionally opt-in and index-specific. Not all distance metrics
// are supported; indexes should return a descriptive error if unsupported.
type ProductQuantizationConfig struct {
	// NumSubvectors is M in PQ literature.
	NumSubvectors int
	// NumCentroids is K in PQ literature (typically 256 for byte codes).
	NumCentroids int
}

// ProductQuantizationEnabler is an optional capability for indexes that can
// build and use PQ codes for faster distance approximations.
//
// Semantics:
//   - Enabling trains a codebook from the current vectors and pre-encodes
//     existing vectors.
//   - After enabling, subsequent inserts/updates should encode new PQ codes.
//   - Disabling removes PQ acceleration (vectors remain unchanged).
type ProductQuantizationEnabler interface {
	EnableProductQuantization(cfg ProductQuantizationConfig) error
	DisableProductQuantization()
	ProductQuantizationEnabled() bool
}
