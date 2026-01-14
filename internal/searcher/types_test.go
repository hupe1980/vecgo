package searcher

import (
	"testing"

	"github.com/hupe1980/vecgo/model"
)

func TestInternalCandidate_ToModel(t *testing.T) {
	ic := InternalCandidate{
		SegmentID: 42,
		RowID:     100,
		Score:     0.95,
		Approx:    true,
	}

	mc := ic.ToModel()

	if mc.Loc.SegmentID != model.SegmentID(42) {
		t.Errorf("SegmentID mismatch: got %d, want 42", mc.Loc.SegmentID)
	}
	if mc.Loc.RowID != model.RowID(100) {
		t.Errorf("RowID mismatch: got %d, want 100", mc.Loc.RowID)
	}
	if mc.Score != 0.95 {
		t.Errorf("Score mismatch: got %f, want 0.95", mc.Score)
	}
	if mc.Approx != true {
		t.Error("Approx mismatch: got false, want true")
	}
}

func TestFromModel(t *testing.T) {
	mc := model.Candidate{
		Loc: model.Location{
			SegmentID: model.SegmentID(42),
			RowID:     model.RowID(100),
		},
		Score:  0.95,
		Approx: true,
	}

	ic := FromModel(mc)

	if ic.SegmentID != 42 {
		t.Errorf("SegmentID mismatch: got %d, want 42", ic.SegmentID)
	}
	if ic.RowID != 100 {
		t.Errorf("RowID mismatch: got %d, want 100", ic.RowID)
	}
	if ic.Score != 0.95 {
		t.Errorf("Score mismatch: got %f, want 0.95", ic.Score)
	}
	if ic.Approx != true {
		t.Error("Approx mismatch: got false, want true")
	}
}

func TestInternalCandidate_RoundTrip(t *testing.T) {
	original := InternalCandidate{
		SegmentID: 123,
		RowID:     456,
		Score:     0.789,
		Approx:    false,
	}

	// Convert to model and back
	mc := original.ToModel()
	roundTripped := FromModel(mc)

	if roundTripped != original {
		t.Errorf("Round-trip failed: got %+v, want %+v", roundTripped, original)
	}
}

func TestInternalCandidate_ZeroValue(t *testing.T) {
	var ic InternalCandidate
	mc := ic.ToModel()

	if mc.Loc.SegmentID != 0 || mc.Loc.RowID != 0 || mc.Score != 0 || mc.Approx != false {
		t.Errorf("Zero value conversion failed: got %+v", mc)
	}

	var mc2 model.Candidate
	ic2 := FromModel(mc2)

	if ic2 != (InternalCandidate{}) {
		t.Errorf("Zero model conversion failed: got %+v", ic2)
	}
}
