package pk

import (
	"fmt"
	"hash/fnv"
	"iter"
	"math/bits"

	"github.com/hupe1980/vecgo/model"
)

const (
	chunkSize = 6
	mask      = (1 << chunkSize) - 1
)

func hashPK(pk model.PK) uint64 {
	if pk.Kind() == model.PKKindUint64 {
		u64, _ := pk.Uint64()
		return u64
	}
	s, _ := pk.StringValue()
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// PersistentIndex is an immutable, persistent index mapping PrimaryKey -> Location.
type PersistentIndex struct {
	root  *node
	count int
}

type node struct {
	bitmap   uint64
	children []*node

	// Leaf data
	key    model.PK
	val    model.Location
	isLeaf bool

	// Collision handling
	collision []entry
}

type entry struct {
	key model.PK
	val model.Location
}

// NewPersistentIndex creates a new empty persistent index.
func NewPersistentIndex() *PersistentIndex {
	return &PersistentIndex{}
}

// Lookup returns the location for the given primary key.
func (idx *PersistentIndex) Lookup(pk model.PK) (model.Location, bool) {
	if idx.root == nil {
		return model.Location{}, false
	}
	return idx.root.lookup(pk, hashPK(pk), 0)
}

func (n *node) lookup(pk model.PK, hash uint64, shift int) (model.Location, bool) {
	if n.isLeaf {
		if len(n.collision) > 0 {
			for _, e := range n.collision {
				if e.key == pk {
					return e.val, true
				}
			}
			return model.Location{}, false
		}
		if n.key == pk {
			return n.val, true
		}
		return model.Location{}, false
	}

	idx := (hash >> shift) & mask
	bit := uint64(1) << idx
	if n.bitmap&bit == 0 {
		return model.Location{}, false
	}

	childIdx := bits.OnesCount64(n.bitmap & (bit - 1))
	return n.children[childIdx].lookup(pk, hash, shift+chunkSize)
}

// Insert returns a new index with the given key-value pair.
func (idx *PersistentIndex) Insert(pk model.PK, loc model.Location) *PersistentIndex {
	newRoot, added := idx.insertRec(idx.root, pk, hashPK(pk), loc, 0)
	newCount := idx.count
	if added {
		newCount++
	}
	return &PersistentIndex{root: newRoot, count: newCount}
}

func (idx *PersistentIndex) insertRec(n *node, pk model.PK, hash uint64, val model.Location, shift int) (*node, bool) {
	if n == nil {
		return &node{
			key:    pk,
			val:    val,
			isLeaf: true,
		}, true
	}

	if n.isLeaf {
		// Check for exact match in collision list
		if len(n.collision) > 0 {
			newCollision := make([]entry, len(n.collision))
			copy(newCollision, n.collision)
			for i, e := range newCollision {
				if e.key == pk {
					newCollision[i].val = val
					return &node{isLeaf: true, collision: newCollision}, false
				}
			}
			// Add to collision
			newCollision = append(newCollision, entry{key: pk, val: val})
			return &node{isLeaf: true, collision: newCollision}, true
		}

		// Check for exact match in single key
		if n.key == pk {
			return &node{
				key:    pk,
				val:    val,
				isLeaf: true,
			}, false
		}

		// Collision or Split
		nHash := hashPK(n.key)
		if nHash == hash {
			// Full hash collision
			return &node{
				isLeaf: true,
				collision: []entry{
					{key: n.key, val: n.val},
					{key: pk, val: val},
				},
			}, true
		}

		// Split leaf
		newInternal := &node{}
		// Insert old leaf
		newInternal, _ = idx.insertRec(newInternal, n.key, nHash, n.val, shift)
		// Insert new value
		return idx.insertRec(newInternal, pk, hash, val, shift)
	}

	// Internal node
	idxBit := (hash >> shift) & mask
	bit := uint64(1) << idxBit
	childIdx := bits.OnesCount64(n.bitmap & (bit - 1))

	var newChild *node
	var added bool

	if n.bitmap&bit != 0 {
		// Update existing child
		if childIdx >= len(n.children) {
			panic(fmt.Sprintf("childIdx %d out of bounds %d", childIdx, len(n.children)))
		}
		newChild, added = idx.insertRec(n.children[childIdx], pk, hash, val, shift+chunkSize)
	} else {
		// Add new child
		newChild = &node{
			key:    pk,
			val:    val,
			isLeaf: true,
		}
		added = true
	}

	if newChild == nil {
		panic("insertRec returned nil child")
	}

	// Copy node
	newNode := &node{
		bitmap: n.bitmap | bit,
		isLeaf: false,
	}

	if n.bitmap&bit != 0 {
		// Replace child
		newNode.children = make([]*node, len(n.children))
		copy(newNode.children, n.children)
		newNode.children[childIdx] = newChild
	} else {
		// Insert child
		newNode.children = make([]*node, len(n.children)+1)
		copy(newNode.children[:childIdx], n.children[:childIdx])
		newNode.children[childIdx] = newChild
		copy(newNode.children[childIdx+1:], n.children[childIdx:])
	}

	return newNode, added
}

// Delete returns a new index with the key removed.
func (idx *PersistentIndex) Delete(pk model.PK) *PersistentIndex {
	newRoot, removed := idx.deleteRec(idx.root, pk, hashPK(pk), 0)
	if !removed {
		return idx
	}
	return &PersistentIndex{root: newRoot, count: idx.count - 1}
}

func (idx *PersistentIndex) deleteRec(n *node, pk model.PK, hash uint64, shift int) (*node, bool) {
	if n == nil {
		return nil, false
	}

	if n.isLeaf {
		if len(n.collision) > 0 {
			// Check collision list
			for i, e := range n.collision {
				if e.key == pk {
					// Remove from collision
					newCollision := make([]entry, 0, len(n.collision)-1)
					newCollision = append(newCollision, n.collision[:i]...)
					newCollision = append(newCollision, n.collision[i+1:]...)

					if len(newCollision) == 0 {
						return nil, true
					}
					if len(newCollision) == 1 {
						// Convert back to single leaf
						return &node{
							key:    newCollision[0].key,
							val:    newCollision[0].val,
							isLeaf: true,
						}, true
					}
					return &node{isLeaf: true, collision: newCollision}, true
				}
			}
			return n, false
		}

		if n.key == pk {
			return nil, true
		}
		return n, false
	}

	idxBit := (hash >> shift) & mask
	bit := uint64(1) << idxBit
	if n.bitmap&bit == 0 {
		return n, false
	}

	childIdx := bits.OnesCount64(n.bitmap & (bit - 1))
	newChild, removed := idx.deleteRec(n.children[childIdx], pk, hash, shift+chunkSize)

	if !removed {
		return n, false
	}

	if newChild == nil {
		// Child removed
		newBitmap := n.bitmap &^ bit
		if newBitmap == 0 {
			return nil, true
		}

		// Optimization: If 1 child remains and it is a leaf, replace this node with that leaf.
		if bits.OnesCount64(newBitmap) == 1 {
			// Find the remaining child
			remainingChild := n.children[0]
			if childIdx == 0 {
				remainingChild = n.children[1]
			}

			if remainingChild.isLeaf {
				return remainingChild, true
			}
		}

		newNode := &node{
			bitmap:   newBitmap,
			isLeaf:   false,
			children: make([]*node, len(n.children)-1),
		}
		copy(newNode.children[:childIdx], n.children[:childIdx])
		copy(newNode.children[childIdx:], n.children[childIdx+1:])
		return newNode, true
	}

	// Child updated
	newNode := &node{
		bitmap:   n.bitmap,
		isLeaf:   false,
		children: make([]*node, len(n.children)),
	}
	copy(newNode.children, n.children)
	newNode.children[childIdx] = newChild
	return newNode, true
}

// Len returns the number of items in the index.
func (idx *PersistentIndex) Len() int {
	return idx.count
}

// Scan returns an iterator over all keys and locations in the index.
func (idx *PersistentIndex) Scan() iter.Seq2[model.PK, model.Location] {
	return func(yield func(model.PK, model.Location) bool) {
		if idx.root == nil {
			return
		}
		idx.root.scan(yield)
	}
}

func (n *node) scan(yield func(model.PK, model.Location) bool) bool {
	if n == nil {
		return true
	}
	if n.isLeaf {
		if !yield(n.key, n.val) {
			return false
		}
		for _, e := range n.collision {
			if !yield(e.key, e.val) {
				return false
			}
		}
		return true
	}
	for _, child := range n.children {
		if child == nil {
			continue
		}
		if !child.scan(yield) {
			return false
		}
	}
	return true
}
