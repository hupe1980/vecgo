package flat

import (
	"bytes"
	"encoding/gob"

	"github.com/hupe1980/vecgo/index"
)

// GobEncode method for Flat.
func (f *Flat) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err := encoder.Encode(f.dimension); err != nil {
		return nil, err
	}

	if err := encoder.Encode(f.nodes); err != nil {
		return nil, err
	}

	if err := encoder.Encode(f.opts); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// GobDecode method for Flat.
func (f *Flat) GobDecode(data []byte) error {
	decoder := gob.NewDecoder(bytes.NewBuffer(data))

	if err := decoder.Decode(&f.dimension); err != nil {
		return err
	}

	if err := decoder.Decode(&f.nodes); err != nil {
		return err
	}

	if err := decoder.Decode(&f.opts); err != nil {
		return err
	}

	f.distanceFunc = index.NewDistanceFunc(f.opts.DistanceType)

	return nil
}
