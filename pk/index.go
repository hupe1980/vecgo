package pk

import (
	"github.com/hupe1980/vecgo/model"
)

// Index is the persistent index mapping PrimaryKey -> Location.
type Index interface {
	Lookup(pk model.PK) (model.Location, bool)
	Upsert(pk model.PK, loc model.Location) error
	Delete(pk model.PK) error
}
