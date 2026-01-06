# SIMD Assembly Generator

Converts C source files with SIMD intrinsics to Go assembly.

## Build

```bash
go build -o generator
```

## Usage

```bash
./generator -arch <arch> -goos <os> [options] <source.c>
```

### Options

- `-v` - Verbose output showing compilation steps
- `-O level` - Optimization level (default: "3")
- `-o dir` - Output directory (default: auto-detected from source path)
- `-arch name` - Target architecture (required, e.g., "amd64", "arm64")
- `-goos name` - Target OS (required, e.g., "linux", "darwin")
- `-go-stubs` - Emit a Go file containing `//go:noescape` declarations derived from parsed C signatures (default: true)
- `-no-tail` - Define `SIMD_NO_TAIL` when compiling C (omit scalar tails in kernels that support it)
- `-D name[=value]` - Additional preprocessor define to pass to clang (repeatable)
- `-allow-relocs` - Allow relocations in the compiled object (unsafe with raw WORD/BYTE emission)
- `-keep-temp` - Keep temporary assembly/object files for debugging

### Examples

```bash
# Generate for AVX (AMD64/Linux)
./generator -arch amd64 -goos linux ../src/floats_avx.c

# Generate for NEON (ARM64/Darwin)
./generator -arch arm64 -goos darwin ../src/floats_neon.c

# Verbose mode
./generator -v -arch amd64 -goos linux ../src/floats_avx.c

# Generate a “bulk-only” kernel (omit scalar tail)
./generator -arch amd64 -goos linux -no-tail ../src/popcount_avx.c
```

## How It Works

1. Compiles C source to assembly using clang with optimization flags
2. Compiles assembly to object file
3. Parses assembly to extract functions, labels, and instructions
4. Uses llvm-objdump to get binary instruction encodings
5. Generates Go assembly with:
   - Proper build tags
   - Go ABI prologue (loads arguments from stack to registers)
   - WORD/BYTE directives containing raw instruction encodings

When `-go-stubs` is enabled, the generator also emits `*_stubs.go` next to the `.s` file.
These stub files contain the `//go:noescape` function declarations for the generated symbols,
so the Go declarations always match the C source signatures.

### Relocations (Important)

The generator emits raw instruction bytes for `.text` only; it does not carry over relocations.
For safety, it fails if the compiled object contains relocations (unless `-allow-relocs` is set).

Common causes of relocations:
- SIMD shuffle masks / constant vectors materialized via `.rodata`.
- Lookup tables defined as `static const` in C.

To keep objects relocation-free:
- Prefer passing lookup tables / masks / constant vectors from Go as pointer parameters.
- If a kernel has a scalar tail loop, either gate it behind `#ifndef SIMD_NO_TAIL` and regenerate
  with `-no-tail`, or compute the tail in Go.

Note: the generator passes `-fno-vectorize -fno-slp-vectorize` to clang to reduce the chance
of literal pools being introduced by auto-vectorization.

## Requirements

- `clang` - C compiler with SIMD support
- `llvm-objdump` - For binary extraction and relocation checking

## Architecture Support

- **ARM64**: Used for NEON kernels
- **AMD64**: Used for AVX/AVX-512 kernels
