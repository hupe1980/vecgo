package pk

import (
	"github.com/hupe1980/vecgo/model"
)

// Index is the persistent index mapping PrimaryKey -> Location.
type Index interface {
	Lookup(pk model.PrimaryKey) (model.Location, bool)
	Upsert(pk model.PrimaryKey, loc model.Location) error
	Delete(pk model.PrimaryKey) error
}
