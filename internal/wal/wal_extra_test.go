package wal

import (
	"errors"
	"encoding/binary"
	"hash/crc32"
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWAL_Extra_Types(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "extra.wal")

	w, err := Open(nil, path, DefaultOptions())
	require.NoError(t, err)

	// String PK Upsert
	rec1 := &Record{
		Type:    RecordTypeUpsert,
		PK:      model.PKString("foo"),
		Vector:  []float32{0.1},
		Payload: []byte("payload"),
	}
	// String PK Delete
	rec2 := &Record{
		Type: RecordTypeDelete,
		PK:   model.PKString("bar"),
	}

	require.NoError(t, w.Append(rec1))
	require.NoError(t, w.Append(rec2))

	// Check Size and Offset
	assert.Greater(t, w.Size(), int64(0))
	
	require.NoError(t, w.Close())

	// Reopen and Verify
	w2, err := Open(nil, path, DefaultOptions())
	require.NoError(t, err)
	defer w2.Close()

	reader, err := w2.Reader()
	require.NoError(t, err)
	
	// Record 1
	r1, err := reader.Next()
	require.NoError(t, err)
	assert.Equal(t, model.PKString("foo"), r1.PK)
	assert.Equal(t, []byte("payload"), r1.Payload)
	assert.Greater(t, reader.Offset(), int64(0))

	// Record 2
	r2, err := reader.Next()
	require.NoError(t, err)
	assert.Equal(t, model.PKString("bar"), r2.PK)
}

func TestRecord_Internal(t *testing.T) {
	// Direct unit tests for record logic
	
	// Size() correctness
	r := &Record{
		Type: RecordTypeDelete,
		PK: model.PKUint64(100),
	}
	// Delete record size: PK Size
	// PK (U64) = 1 (kind) + 8 (u64) = 9
	// Delete payload = 9
	assert.Equal(t, 26, r.Size())

	r2 := &Record{
		Type: RecordTypeDelete,
		PK: model.PKString("abc"),
	}
	// PK (String) = 1 (kind) + 4 (len) + 3 (abc) = 8
	// Wait, is it?
	// pkSize func: 1 + 4 + len
	// 1 + 4 + 3 = 8.
	assert.Equal(t, 8, int(pkSize(r2.PK)))
	assert.Equal(t, 25, r2.Size())
}

func TestWAL_Corrupt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corrupt.wal")
	
	// Create valid wal
	w, err := Open(nil, path, DefaultOptions())
	require.NoError(t, err)
	w.Append(&Record{Type: RecordTypeDelete, PK: model.PKUint64(1)})
	w.Close()
	
	// Truncate it to corrupt last record
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	require.NoError(t, err)
	fi, _ := f.Stat()
	f.Truncate(fi.Size() - 1)
	f.Close()
	
	// Reader should handle it (return EOF or error?)
	w2, err := Open(nil, path, DefaultOptions())
	require.NoError(t, err)
	defer w2.Close()
	
	reader, err := w2.Reader()
	require.NoError(t, err)
	
	_, err = reader.Next()
	assert.Error(t, err)
	// Could be checksum error or short read
}

func TestWAL_OpenError(t *testing.T) {
        dir := t.TempDir()
        // Create a read-only directory inside
        roDir := filepath.Join(dir, "readonly")
        err := os.Mkdir(roDir, 0500) // Read + Execute (so we can enter it, but not write files)
        require.NoError(t, err)

        path := filepath.Join(roDir, "test.wal")
        _, err = Open(nil, path, DefaultOptions())
        assert.Error(t, err)
}

func TestRecord_DecodeErrors(t *testing.T) {
        // 1. Short Read Header
        shortData := []byte{0x00, 0x01}
        _, _, err := Decode(bytes.NewReader(shortData))
        assert.Error(t, err)

        // 2. Invalid CRC
        validRec := &Record{Type: RecordTypeDelete, PK: model.PKUint64(1)}
        buf := new(bytes.Buffer)
        validRec.Encode(buf)
        data := buf.Bytes()
        // corrupt crc (first 4 bytes)
        data[0]++ 
        _, _, err = Decode(bytes.NewReader(data))
        assert.Equal(t, ErrInvalidCRC, err)

        // 3. Invalid Type
        header := make([]byte, 13)
        header[0] = 99 // Invalid type
        binary.LittleEndian.PutUint64(header[1:], 1)
        binary.LittleEndian.PutUint32(header[9:], 0) // zero length payload
        
        crc := crc32.NewIEEE()
        crc.Write(header)
        // payload empty (0 length)
        checksum := crc.Sum32()
        
        buf2 := new(bytes.Buffer)
        binary.Write(buf2, binary.LittleEndian, checksum)
        buf2.Write(header)
        
        _, _, err = Decode(buf2)
        assert.Equal(t, ErrInvalidType, err)

        // 4. Malformed Upsert Payload (Short Read inside parseUpsert)
        // We construct a payload that has valid CRC but invalid internal structure
        payload := make([]byte, 0)
        scratch := make([]byte, 8)
        
        // PK (U64) - valid
        payload = append(payload, byte(model.PKKindUint64))
        binary.LittleEndian.PutUint64(scratch, 1)
        payload = append(payload, scratch...)
        
        // Dim = 1000 (indicates 4000 bytes following)
        binary.LittleEndian.PutUint32(scratch[:4], 1000)
        payload = append(payload, scratch[:4]...)
        // Missing vector bytes...

        // Header
        h := make([]byte, 13)
        h[0] = byte(RecordTypeUpsert)
        binary.LittleEndian.PutUint32(h[9:], uint32(len(payload))) // Length matches actual payload size
        
        c := crc32.NewIEEE()
        c.Write(h)
        c.Write(payload)
        sum := c.Sum32() // Valid checksum for this payload
        
        buf4 := new(bytes.Buffer)
        binary.Write(buf4, binary.LittleEndian, sum)
        buf4.Write(h)
        buf4.Write(payload)
        
        _, _, err = Decode(buf4)
        assert.Equal(t, ErrShortRead, err)
}

type FailWriter struct {
        FailAt int
        Count  int
}

func (fw *FailWriter) Write(p []byte) (int, error) {
        if fw.Count >= fw.FailAt {
                return 0, errors.New("write error")
        }
        if fw.Count+len(p) > fw.FailAt {
                n := fw.FailAt - fw.Count
                fw.Count = fw.FailAt
                return n, errors.New("write error")
        }
        fw.Count += len(p)
        return len(p), nil
}

func TestEncode_Errors(t *testing.T) {
        r := &Record{
                Type:     RecordTypeUpsert,
                PK:       model.PKString("test"), // string PK to hit complex path
                Vector:   []float32{1.0, 2.0},
                Metadata: []byte("meta"),
                Payload:  []byte("payload"),
        }
        
        // Try failing at different offsets
        for i := 0; i < 200; i++ {
            fw := &FailWriter{FailAt: i}
            if err := r.Encode(fw); err == nil {
                break
            }
        }
        
        // Also test RecordTypeDelete encode errors
        r2 := &Record{
             Type: RecordTypeDelete,
             PK: model.PKUint64(123),
        }
        for i := 0; i < 50; i++ {
            fw := &FailWriter{FailAt: i}
            if err := r2.Encode(fw); err == nil {
                break
            }
        }
}
