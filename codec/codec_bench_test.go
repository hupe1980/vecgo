package codec

import (
	"testing"

	"github.com/hupe1980/vecgo/metadata"
)

type benchChild struct {
	K string `json:"k"`
	V int64  `json:"v"`
}

type benchPayload struct {
	ID       uint64            `json:"id"`
	Title    string            `json:"title"`
	Score    float64           `json:"score"`
	Tags     []string          `json:"tags"`
	Attrs    map[string]string `json:"attrs"`
	Flags    []bool            `json:"flags"`
	Children []benchChild      `json:"children"`
}

func benchmarkCodecMarshal(b *testing.B, c Codec, v any) {
	b.Helper()
	b.ReportAllocs()

	warm, err := c.Marshal(v)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(warm)))

	var sink []byte
	b.ResetTimer()
	for b.Loop() {
		out, err := c.Marshal(v)
		if err != nil {
			b.Fatal(err)
		}
		sink = out
	}
	_ = sink
}

func benchmarkCodecUnmarshal[T any](b *testing.B, c Codec, data []byte, dst *T) {
	b.Helper()
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))

	var v T
	b.ResetTimer()
	for b.Loop() {
		if err := c.Unmarshal(data, &v); err != nil {
			b.Fatal(err)
		}
	}
	if dst != nil {
		*dst = v
	}
}

func BenchmarkCodec_Marshal_Payload(b *testing.B) {
	payload := benchPayload{
		ID:    123456789,
		Title: "hello vecgo",
		Score: 0.12345,
		Tags:  []string{"a", "b", "c", "d", "e"},
		Attrs: map[string]string{
			"kind":  "bench",
			"owner": "hupe1980",
			"repo":  "vecgo",
			"lang":  "go",
		},
		Flags: []bool{true, false, true, true, false, false, true},
		Children: []benchChild{
			{K: "x", V: 1},
			{K: "y", V: 2},
			{K: "z", V: 3},
		},
	}

	b.Run("stdlib", func(b *testing.B) { benchmarkCodecMarshal(b, JSON{}, payload) })
	b.Run("go-json", func(b *testing.B) { benchmarkCodecMarshal(b, GoJSON{}, payload) })
}

func BenchmarkCodec_Unmarshal_Payload(b *testing.B) {
	payload := benchPayload{
		ID:    123456789,
		Title: "hello vecgo",
		Score: 0.12345,
		Tags:  []string{"a", "b", "c", "d", "e"},
		Attrs: map[string]string{
			"kind":  "bench",
			"owner": "hupe1980",
			"repo":  "vecgo",
			"lang":  "go",
		},
		Flags: []bool{true, false, true, true, false, false, true},
		Children: []benchChild{
			{K: "x", V: 1},
			{K: "y", V: 2},
			{K: "z", V: 3},
		},
	}

	jsonData := MustMarshal(JSON{}, payload)

	b.Run("stdlib", func(b *testing.B) {
		var sink benchPayload
		benchmarkCodecUnmarshal(b, JSON{}, jsonData, &sink)
		_ = sink
	})
	b.Run("go-json", func(b *testing.B) {
		var sink benchPayload
		benchmarkCodecUnmarshal(b, GoJSON{}, jsonData, &sink)
		_ = sink
	})
}

func BenchmarkCodec_Marshal_Metadata(b *testing.B) {
	m := metadata.Metadata{
		"tenant":  metadata.String("acme"),
		"doc_id":  metadata.Int(42),
		"rating":  metadata.Float(4.75),
		"active":  metadata.Bool(true),
		"tags":    metadata.Array([]metadata.Value{metadata.String("a"), metadata.String("b"), metadata.String("c")}),
		"numbers": metadata.Array([]metadata.Value{metadata.Int(1), metadata.Int(2), metadata.Int(3), metadata.Int(4)}),
	}

	b.Run("stdlib", func(b *testing.B) { benchmarkCodecMarshal(b, JSON{}, m) })
	b.Run("go-json", func(b *testing.B) { benchmarkCodecMarshal(b, GoJSON{}, m) })
}

func BenchmarkCodec_Unmarshal_Metadata(b *testing.B) {
	m := metadata.Metadata{
		"tenant":  metadata.String("acme"),
		"doc_id":  metadata.Int(42),
		"rating":  metadata.Float(4.75),
		"active":  metadata.Bool(true),
		"tags":    metadata.Array([]metadata.Value{metadata.String("a"), metadata.String("b"), metadata.String("c")}),
		"numbers": metadata.Array([]metadata.Value{metadata.Int(1), metadata.Int(2), metadata.Int(3), metadata.Int(4)}),
	}

	jsonData := MustMarshal(JSON{}, m)

	b.Run("stdlib", func(b *testing.B) {
		var sink metadata.Metadata
		benchmarkCodecUnmarshal(b, JSON{}, jsonData, &sink)
		_ = sink
	})
	b.Run("go-json", func(b *testing.B) {
		var sink metadata.Metadata
		benchmarkCodecUnmarshal(b, GoJSON{}, jsonData, &sink)
		_ = sink
	})
}
