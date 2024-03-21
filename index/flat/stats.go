package flat

import "fmt"

// Stats prints statistics about flat
func (f *Flat) Stats() {
	fmt.Println("Options:")
	fmt.Printf("\tDistanceType = %s\n", f.opts.DistanceType)

	fmt.Println("Parameters:")
	fmt.Printf("\tdimension = %d\n", f.dimension)
}
