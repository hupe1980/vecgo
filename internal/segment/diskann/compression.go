package diskann

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"sync"

	"github.com/klauspost/compress/zstd"
	"github.com/pierrec/lz4/v4"
)

// CompressionType defines the compression algorithm used.
type CompressionType uint8

const (
	// CompressionNone indicates no compression.
	CompressionNone CompressionType = 0
	// CompressionLZ4 indicates LZ4 block compression (fast, good for hot data).
	CompressionLZ4 CompressionType = 1
	// CompressionZSTD indicates ZSTD block compression (better ratio, good for cold data).
	CompressionZSTD CompressionType = 2
)

// ZSTD encoder/decoder pools for efficiency
var (
	zstdEncoderPool sync.Pool
	zstdDecoderPool sync.Pool
)

func getZstdEncoder() *zstd.Encoder {
	if v := zstdEncoderPool.Get(); v != nil {
		return v.(*zstd.Encoder)
	}
	// Level 3 balances compression ratio vs speed
	enc, _ := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
	return enc
}

func putZstdEncoder(enc *zstd.Encoder) {
	zstdEncoderPool.Put(enc)
}

func getZstdDecoder() *zstd.Decoder {
	if v := zstdDecoderPool.Get(); v != nil {
		return v.(*zstd.Decoder)
	}
	dec, _ := zstd.NewReader(nil)
	return dec
}

func putZstdDecoder(dec *zstd.Decoder) {
	zstdDecoderPool.Put(dec)
}

// BlockHeader is the header for a compressed block.
// Format: [UncompressedSize uint32][CompressedSize uint32][Data...]
// If CompressedSize == 0, the block is stored uncompressed.
type BlockHeader struct {
	UncompressedSize uint32
	CompressedSize   uint32 // 0 means uncompressed
}

const blockHeaderSize = 8

// compressBlock compresses a block using the specified algorithm.
// Returns the compressed data with header, or original data if compression doesn't help.
func compressBlock(data []byte, compressionType CompressionType) ([]byte, error) {
	if compressionType == CompressionNone || len(data) == 0 {
		return data, nil
	}

	var compressed []byte
	var err error

	switch compressionType {
	case CompressionLZ4:
		compressed, err = compressBlockLZ4(data)
	case CompressionZSTD:
		compressed, err = compressBlockZSTD(data)
	default:
		return data, nil
	}

	if err != nil {
		return nil, err
	}

	// If compression doesn't help (ratio > 0.9), store uncompressed
	if len(compressed) == 0 || float64(len(compressed)) > float64(len(data))*0.9 {
		// Store uncompressed with header
		result := make([]byte, blockHeaderSize+len(data))
		binary.LittleEndian.PutUint32(result[0:], uint32(len(data)))
		binary.LittleEndian.PutUint32(result[4:], 0) // 0 = uncompressed
		copy(result[blockHeaderSize:], data)
		return result, nil
	}

	// Store compressed with header
	result := make([]byte, blockHeaderSize+len(compressed))
	binary.LittleEndian.PutUint32(result[0:], uint32(len(data)))
	binary.LittleEndian.PutUint32(result[4:], uint32(len(compressed)))
	copy(result[blockHeaderSize:], compressed)
	return result, nil
}

// compressBlockLZ4 compresses data using LZ4.
func compressBlockLZ4(data []byte) ([]byte, error) {
	maxCompressedSize := lz4.CompressBlockBound(len(data))
	compressed := make([]byte, maxCompressedSize)

	n, err := lz4.CompressBlock(data, compressed, nil)
	if err != nil {
		return nil, err
	}

	if n == 0 {
		return nil, nil // Incompressible
	}

	return compressed[:n], nil
}

// compressBlockZSTD compresses data using ZSTD.
func compressBlockZSTD(data []byte) ([]byte, error) {
	enc := getZstdEncoder()
	defer putZstdEncoder(enc)

	return enc.EncodeAll(data, nil), nil
}

// decompressBlock decompresses a block using the appropriate algorithm.
// The compression type is auto-detected from the data format.
func decompressBlock(data []byte, compressionType CompressionType) ([]byte, error) {
	if len(data) < blockHeaderSize {
		return nil, errors.New("block too small for header")
	}

	uncompressedSize := binary.LittleEndian.Uint32(data[0:])
	compressedSize := binary.LittleEndian.Uint32(data[4:])

	if compressedSize == 0 {
		// Uncompressed block
		if uint32(len(data)) < blockHeaderSize+uncompressedSize {
			return nil, errors.New("block data too small")
		}
		return data[blockHeaderSize : blockHeaderSize+uncompressedSize], nil
	}

	// Compressed block
	if uint32(len(data)) < blockHeaderSize+compressedSize {
		return nil, errors.New("compressed block data too small")
	}

	compressedData := data[blockHeaderSize : blockHeaderSize+compressedSize]
	result := make([]byte, uncompressedSize)

	switch compressionType {
	case CompressionLZ4:
		n, err := lz4.UncompressBlock(compressedData, result)
		if err != nil {
			return nil, err
		}
		if uint32(n) != uncompressedSize {
			return nil, errors.New("decompressed size mismatch")
		}
		return result, nil

	case CompressionZSTD:
		dec := getZstdDecoder()
		defer putZstdDecoder(dec)

		decoded, err := dec.DecodeAll(compressedData, result[:0])
		if err != nil {
			return nil, err
		}
		if uint32(len(decoded)) != uncompressedSize {
			return nil, errors.New("decompressed size mismatch")
		}
		return decoded, nil

	default:
		// Try LZ4 as fallback for backward compatibility
		n, err := lz4.UncompressBlock(compressedData, result)
		if err != nil {
			return nil, err
		}
		if uint32(n) != uncompressedSize {
			return nil, errors.New("decompressed size mismatch")
		}
		return result, nil
	}
}

// CompressedBlockWriter writes compressed blocks to an underlying writer.
type CompressedBlockWriter struct {
	w               io.Writer
	compressionType CompressionType
	blockSize       int
	buffer          *bytes.Buffer
	written         int64
}

// NewCompressedBlockWriter creates a new compressed block writer.
func NewCompressedBlockWriter(w io.Writer, compressionType CompressionType, blockSize int) *CompressedBlockWriter {
	if blockSize <= 0 {
		blockSize = 256 * 1024 // 256KB default block size
	}
	return &CompressedBlockWriter{
		w:               w,
		compressionType: compressionType,
		blockSize:       blockSize,
		buffer:          bytes.NewBuffer(make([]byte, 0, blockSize)),
	}
}

// Write writes data to the buffer, flushing blocks as needed.
func (c *CompressedBlockWriter) Write(p []byte) (int, error) {
	total := 0
	for len(p) > 0 {
		space := c.blockSize - c.buffer.Len()
		if space <= 0 {
			if err := c.FlushBlock(); err != nil {
				return total, err
			}
			space = c.blockSize
		}

		toWrite := len(p)
		if toWrite > space {
			toWrite = space
		}

		n, err := c.buffer.Write(p[:toWrite])
		if err != nil {
			return total, err
		}
		total += n
		p = p[n:]
	}
	return total, nil
}

// FlushBlock compresses and writes the current block.
func (c *CompressedBlockWriter) FlushBlock() error {
	if c.buffer.Len() == 0 {
		return nil
	}

	compressed, err := compressBlock(c.buffer.Bytes(), c.compressionType)
	if err != nil {
		return err
	}

	n, err := c.w.Write(compressed)
	if err != nil {
		return err
	}
	c.written += int64(n)
	c.buffer.Reset()
	return nil
}

// Flush writes any remaining buffered data.
func (c *CompressedBlockWriter) Flush() error {
	return c.FlushBlock()
}

// BytesWritten returns the total compressed bytes written.
func (c *CompressedBlockWriter) BytesWritten() int64 {
	return c.written
}

// CompressedBlockReader reads compressed blocks from an underlying reader.
type CompressedBlockReader struct {
	data            []byte
	offset          int64
	buffer          []byte
	bufStart        int
	bufEnd          int
	compressionType CompressionType
}

// NewCompressedBlockReader creates a reader for compressed blocks.
func NewCompressedBlockReader(data []byte, startOffset int64, compressionType CompressionType) *CompressedBlockReader {
	return &CompressedBlockReader{
		data:            data,
		offset:          startOffset,
		compressionType: compressionType,
	}
}

// ReadBlock reads and decompresses the next block.
func (c *CompressedBlockReader) ReadBlock() ([]byte, error) {
	if int(c.offset)+blockHeaderSize > len(c.data) {
		return nil, io.EOF
	}

	uncompressedSize := binary.LittleEndian.Uint32(c.data[c.offset:])
	compressedSize := binary.LittleEndian.Uint32(c.data[c.offset+4:])

	var blockData []byte
	if compressedSize == 0 {
		// Uncompressed
		dataSize := uncompressedSize
		if int(c.offset)+blockHeaderSize+int(dataSize) > len(c.data) {
			return nil, errors.New("block extends beyond data")
		}
		blockData = c.data[c.offset+blockHeaderSize : c.offset+blockHeaderSize+int64(dataSize)]
		c.offset += blockHeaderSize + int64(dataSize)
	} else {
		// Compressed
		if int(c.offset)+blockHeaderSize+int(compressedSize) > len(c.data) {
			return nil, errors.New("compressed block extends beyond data")
		}

		compressedData := c.data[c.offset+blockHeaderSize : c.offset+blockHeaderSize+int64(compressedSize)]
		result := make([]byte, uncompressedSize)

		switch c.compressionType {
		case CompressionZSTD:
			dec := getZstdDecoder()
			defer putZstdDecoder(dec)

			decoded, err := dec.DecodeAll(compressedData, result[:0])
			if err != nil {
				return nil, err
			}
			if uint32(len(decoded)) != uncompressedSize {
				return nil, errors.New("decompressed size mismatch")
			}
			blockData = decoded

		default: // LZ4 or fallback
			n, err := lz4.UncompressBlock(compressedData, result)
			if err != nil {
				return nil, err
			}
			if uint32(n) != uncompressedSize {
				return nil, errors.New("decompressed size mismatch")
			}
			blockData = result
		}

		c.offset += blockHeaderSize + int64(compressedSize)
	}

	return blockData, nil
}

// DecompressAll reads all compressed blocks and returns the full decompressed data.
func DecompressAll(data []byte, startOffset int64, compressionType CompressionType) ([]byte, error) {
	reader := NewCompressedBlockReader(data, startOffset, compressionType)
	var result []byte

	for {
		block, err := reader.ReadBlock()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		result = append(result, block...)
	}

	return result, nil
}
