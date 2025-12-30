package persistence

import (
	"encoding/binary"
	"io"
)

// HNSWMetadata contains HNSW index metadata (28 bytes).
type HNSWMetadata struct {
	MaxLayers    uint16   // Maximum number of layers
	M            uint16   // Maximum connections per node
	Ml           float32  // Layer multiplier
	EntryPoint   uint64   // Entry point node ID
	DistanceFunc uint8    // index.DistanceType (e.g. 0=SquaredL2, 1=CosineSimilarity)
	Flags        uint8    // Metadata flags (bit 0: vectors normalized)
	Padding      [10]byte // Reserved for future use
}

// WriteHNSWMetadata writes HNSW metadata.
func WriteHNSWMetadata(w io.Writer, meta *HNSWMetadata) error {
	return binary.Write(w, binary.LittleEndian, meta)
}

// ReadHNSWMetadata reads HNSW metadata.
func ReadHNSWMetadata(r io.Reader) (*HNSWMetadata, error) {
	var meta HNSWMetadata
	if err := binary.Read(r, binary.LittleEndian, &meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

// WriteConnections writes multi-layer connection structure.
// Format: [layer0_count][layer0_ids...][layer1_count][layer1_ids...]...
func WriteConnections(w io.Writer, connections [][]uint64) error {
	writer := NewBinaryIndexWriter(w)
	for _, layer := range connections {
		// Write count as uint32
		count := uint32(len(layer))
		if err := binary.Write(w, binary.LittleEndian, count); err != nil {
			return err
		}
		// Write connection IDs
		if count > 0 {
			if err := writer.WriteUint64Slice(layer); err != nil {
				return err
			}
		}
	}
	return nil
}

// ReadConnections reads multi-layer connection structure.
func ReadConnections(r io.Reader, layerCount int) ([][]uint64, error) {
	reader := NewBinaryIndexReader(r)
	connections := make([][]uint64, layerCount)
	for i := 0; i < layerCount; i++ {
		var count uint32
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		if count > 0 {
			layer, err := reader.ReadUint64Slice(int(count))
			if err != nil {
				return nil, err
			}
			connections[i] = layer
		}
	}
	return connections, nil
}
