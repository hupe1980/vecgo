# 🧬🔍🗄️ VecGo

VecGo is a Go library designed for efficient vector indexing and searching, emphasizing approximate nearest neighbor search using the HNSW (Hierarchical Navigable Small World) algorithm. It offers a versatile and user-friendly interface for managing and querying vast collections of high-dimensional vectors.

:warning: This is experimental and subject to breaking changes.

## Features

- HNSW algorithm implementation for approximate nearest neighbor search.
- Support for brute-force and heuristic search methods.
- Customizable options for memory usage and search performance.
- Efficient handling of high-dimensional vectors.
- Embeddable vector store for seamless integration with Go applications.

## Usage

Here's a basic example demonstrating how to perform nearest neighbor search with VecGo:


```go
package main

import (
	"fmt"
	"log"

	"github.com/hupe1980/vecgo"
)

func main() {
	vg := vecgo.New[string](3)

	_, err := vg.Insert(&vecgo.VectorWithData[string]{
		Vector: []float32{1.0, 2.0, 2.5},
		Data:   "Hello World!",
	})
	if err != nil {
		log.Fatal(err)
	}

	k := 5
	result, err := vg.KNNSearch([]float32{1.0, 2.0, 2.5}, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result[0].Data)
}
```

For more information on each method, please refer to the GoDoc documentation.

## Contributing
Contributions to Vecgo are welcome! Feel free to open issues for bug reports, feature requests, or submit pull requests with improvements.

## License
Vecgo is licensed under the MIT License. See the LICENSE file for details.