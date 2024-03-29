# 🧬🔍🗄️ Vecgo
![Build Status](https://github.com/hupe1980/vecgo/workflows/build/badge.svg) 
[![Go Reference](https://pkg.go.dev/badge/github.com/hupe1980/vecgo.svg)](https://pkg.go.dev/github.com/hupe1980/vecgo)
[![goreportcard](https://goreportcard.com/badge/github.com/hupe1980/vecgo)](https://goreportcard.com/report/github.com/hupe1980/vecgo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Vecgo is a Go library designed for efficient vector indexing and searching, supporting various index types and emphasizing approximate nearest neighbor search. It provides a versatile and user-friendly interface for managing and querying vast collections of high-dimensional vectors.

:warning: This is experimental and subject to breaking changes.

## Features

- Support for multiple index types, including flat and HNSW (Hierarchical Navigable Small World) algorithms.
- Customizable options for memory usage and search performance.
- Efficient handling of high-dimensional vectors.
- Embeddable vector store for seamless integration with Go applications.

## Usage

Here's a basic example demonstrating how to perform nearest neighbor search with Vecgo:


```go
package main

import (
	"fmt"
	"log"

	"github.com/hupe1980/vecgo"
)

func main() {
	vg := vecgo.NewHNSW[string]()
	// vg := vecgo.NewFlat[string]()

	_, err := vg.Insert(vecgo.VectorWithData[string]{
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