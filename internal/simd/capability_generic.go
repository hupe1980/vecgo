//go:build !arm64 && !amd64

package simd

func init() {
	// No SIMD features on this platform
	initCapabilities()
}
