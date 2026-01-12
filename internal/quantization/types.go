package quantization

// Type represents the type of quantization method.
type Type int

const (
	TypeNone Type = iota
	TypePQ
	TypeOPQ
	TypeSQ8
	TypeBQ
	TypeRaBitQ
	TypeINT4
)

// String returns the string representation of the quantization type.
func (t Type) String() string {
	switch t {
	case TypeNone:
		return "None"
	case TypePQ:
		return "PQ"
	case TypeOPQ:
		return "OPQ"
	case TypeSQ8:
		return "SQ8"
	case TypeBQ:
		return "BQ"
	case TypeRaBitQ:
		return "RaBitQ"
	case TypeINT4:
		return "INT4"
	default:
		return "Unknown"
	}
}
