package diskann

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompressBlock_LZ4(t *testing.T) {
	// Test with compressible data (repeated patterns)
	data := bytes.Repeat([]byte("hello world! "), 1000)

	compressed, err := compressBlock(data, CompressionLZ4)
	require.NoError(t, err)

	// Should be significantly smaller
	assert.Less(t, len(compressed), len(data)/2, "LZ4 should compress repeated data well")

	// Decompress and verify
	decompressed, err := decompressBlock(compressed, CompressionLZ4)
	require.NoError(t, err)
	assert.Equal(t, data, decompressed)
}

func TestCompressBlock_ZSTD(t *testing.T) {
	// Test with compressible data (repeated patterns)
	data := bytes.Repeat([]byte("hello world! "), 1000)

	compressed, err := compressBlock(data, CompressionZSTD)
	require.NoError(t, err)

	// Should be significantly smaller
	assert.Less(t, len(compressed), len(data)/2, "ZSTD should compress repeated data well")

	// Decompress and verify
	decompressed, err := decompressBlock(compressed, CompressionZSTD)
	require.NoError(t, err)
	assert.Equal(t, data, decompressed)
}

func TestCompressBlock_NoCompression(t *testing.T) {
	data := []byte("small data that won't benefit from compression")

	result, err := compressBlock(data, CompressionNone)
	require.NoError(t, err)

	// Should be unchanged
	assert.Equal(t, data, result)
}

func TestCompressBlock_IncompressibleData(t *testing.T) {
	// Random-ish data that doesn't compress well
	data := make([]byte, 1000)
	for i := range data {
		data[i] = byte(i * 17 % 256)
	}

	compressed, err := compressBlock(data, CompressionLZ4)
	require.NoError(t, err)

	// Should still be decompressible
	decompressed, err := decompressBlock(compressed, CompressionLZ4)
	require.NoError(t, err)
	assert.Equal(t, data, decompressed)
}

func TestCompressedBlockWriter(t *testing.T) {
	var buf bytes.Buffer
	w := NewCompressedBlockWriter(&buf, CompressionLZ4, 1024) // 1KB blocks

	// Write more than one block
	data := bytes.Repeat([]byte("test data for compression "), 100)
	n, err := w.Write(data)
	require.NoError(t, err)
	assert.Equal(t, len(data), n)

	err = w.Flush()
	require.NoError(t, err)

	// Compressed output should exist
	assert.Greater(t, buf.Len(), 0)
	assert.Less(t, buf.Len(), len(data), "compressed should be smaller for repetitive data")
}

func TestDecompressAll(t *testing.T) {
	// Create compressed data
	var buf bytes.Buffer
	w := NewCompressedBlockWriter(&buf, CompressionLZ4, 256) // Small blocks

	original := bytes.Repeat([]byte("compress me! "), 500)
	_, err := w.Write(original)
	require.NoError(t, err)
	err = w.Flush()
	require.NoError(t, err)

	// Decompress all
	decompressed, err := DecompressAll(buf.Bytes(), 0, CompressionLZ4)
	require.NoError(t, err)
	assert.Equal(t, original, decompressed)
}

func TestDecompressAll_ZSTD(t *testing.T) {
	// Create ZSTD compressed data
	var buf bytes.Buffer
	w := NewCompressedBlockWriter(&buf, CompressionZSTD, 256) // Small blocks

	original := bytes.Repeat([]byte("compress me with zstd! "), 500)
	_, err := w.Write(original)
	require.NoError(t, err)
	err = w.Flush()
	require.NoError(t, err)

	// Decompress all
	decompressed, err := DecompressAll(buf.Bytes(), 0, CompressionZSTD)
	require.NoError(t, err)
	assert.Equal(t, original, decompressed)
}

func TestCompressedBlockReader(t *testing.T) {
	// Create multiple compressed blocks
	var buf bytes.Buffer
	w := NewCompressedBlockWriter(&buf, CompressionLZ4, 100) // Very small blocks

	original := bytes.Repeat([]byte("block data "), 50)
	_, err := w.Write(original)
	require.NoError(t, err)
	err = w.Flush()
	require.NoError(t, err)

	// Read blocks one by one
	reader := NewCompressedBlockReader(buf.Bytes(), 0, CompressionLZ4)
	var reconstructed []byte

	for {
		block, err := reader.ReadBlock()
		if err != nil {
			break // EOF
		}
		reconstructed = append(reconstructed, block...)
	}

	assert.Equal(t, original, reconstructed)
}

func BenchmarkCompressBlock(b *testing.B) {
	data := bytes.Repeat([]byte("benchmark data for compression testing "), 1000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = compressBlock(data, CompressionLZ4)
	}
}

func BenchmarkCompressBlock_ZSTD(b *testing.B) {
	data := bytes.Repeat([]byte("benchmark data for compression testing "), 1000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = compressBlock(data, CompressionZSTD)
	}
}

func BenchmarkDecompressBlock(b *testing.B) {
	data := bytes.Repeat([]byte("benchmark data for compression testing "), 1000)
	compressed, _ := compressBlock(data, CompressionLZ4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = decompressBlock(compressed, CompressionLZ4)
	}
}

func BenchmarkDecompressBlock_ZSTD(b *testing.B) {
	data := bytes.Repeat([]byte("benchmark data for compression testing "), 1000)
	compressed, _ := compressBlock(data, CompressionZSTD)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = decompressBlock(compressed, CompressionZSTD)
	}
}
