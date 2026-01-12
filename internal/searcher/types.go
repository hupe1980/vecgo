package searcher

import "github.com/hupe1980/vecgo/model"

// InternalCandidate is a lightweight representation of a candidate used during search.
// optimized for memory layout (16 bytes) to reduce copying overhead throughout the heap operations.
type InternalCandidate struct {
	SegmentID uint32  // 4 bytes
	RowID     uint32  // 4 bytes
	Score     float32 // 4 bytes
	Approx    bool    // 1 byte
	// Total 13 -> padded to 16 bytes
}

// ToModel converts InternalCandidate to model.Candidate
func (c InternalCandidate) ToModel() model.Candidate {
	return model.Candidate{
		Loc: model.Location{
			SegmentID: model.SegmentID(c.SegmentID),
			RowID:     model.RowID(c.RowID),
		},
		Score:  c.Score,
		Approx: c.Approx,
	}
}

// FromModel converts model.Candidate to InternalCandidate
func FromModel(c model.Candidate) InternalCandidate {
	return InternalCandidate{
		SegmentID: uint32(c.Loc.SegmentID),
		RowID:     uint32(c.Loc.RowID),
		Score:     c.Score,
		Approx:    c.Approx,
	}
}
