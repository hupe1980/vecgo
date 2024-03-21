package hnsw

import (
	"bytes"
	"encoding/gob"

	"github.com/hupe1980/vecgo/index"
)

// GobEncode method for HNSW.
func (h *HNSW) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err := encoder.Encode(h.dimension); err != nil {
		return nil, err
	}

	if err := encoder.Encode(h.mmax); err != nil {
		return nil, err
	}

	if err := encoder.Encode(h.mmax0); err != nil {
		return nil, err
	}

	if err := encoder.Encode(h.ml); err != nil {
		return nil, err
	}

	if err := encoder.Encode(h.ep); err != nil {
		return nil, err
	}

	if err := encoder.Encode(h.maxLevel); err != nil {
		return nil, err
	}

	if err := encoder.Encode(h.nodes); err != nil {
		return nil, err
	}

	if err := encoder.Encode(h.opts); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// GobDecode method for HNSW.
func (h *HNSW) GobDecode(data []byte) error {
	decoder := gob.NewDecoder(bytes.NewBuffer(data))

	if err := decoder.Decode(&h.dimension); err != nil {
		return err
	}

	if err := decoder.Decode(&h.mmax); err != nil {
		return err
	}

	if err := decoder.Decode(&h.mmax0); err != nil {
		return err
	}

	if err := decoder.Decode(&h.ml); err != nil {
		return err
	}

	if err := decoder.Decode(&h.ep); err != nil {
		return err
	}

	if err := decoder.Decode(&h.maxLevel); err != nil {
		return err
	}

	if err := decoder.Decode(&h.nodes); err != nil {
		return err
	}

	if err := decoder.Decode(&h.opts); err != nil {
		return err
	}

	h.distanceFunc = index.NewDistanceFunc(h.opts.DistanceType)

	return nil
}
