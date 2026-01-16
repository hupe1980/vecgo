//go:build amd64

package simd

import "golang.org/x/sys/cpu"

func init() {
	hasAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	hasAVX512F = cpu.X86.HasAVX512F
	hasAVX512BW = cpu.X86.HasAVX512BW
	initCapabilities()
}
