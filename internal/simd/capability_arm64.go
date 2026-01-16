//go:build arm64

package simd

import "golang.org/x/sys/cpu"

func init() {
	hasASIMD = cpu.ARM64.HasASIMD
	hasSVE2 = cpu.ARM64.HasSVE2
	initCapabilities()
}
