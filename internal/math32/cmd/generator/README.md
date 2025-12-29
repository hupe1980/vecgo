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
- `-pkg name` - Package name (default: "math32")
- `-arch name` - Target architecture (required, e.g., "amd64", "arm64")
- `-goos name` - Target OS (required, e.g., "linux", "darwin")

### Examples

```bash
# Generate for AVX (AMD64/Linux)
./generator -arch amd64 -goos linux ../src/floats_avx.c

# Generate for NEON (ARM64/Darwin)
./generator -arch arm64 -goos darwin ../src/floats_neon.c

# Verbose mode
./generator -v -arch amd64 -goos linux ../src/floats_avx.c
```

## How It Works

1. Compiles C source to assembly using clang with optimization flags
2. Compiles assembly to object file
3. Parses assembly to extract functions, labels, and instructions
4. Uses llvm-objdump to get binary instruction encodings
5. Generates Go assembly with:
   - Proper build tags
   - Go ABI prologue (loads arguments from stack to registers)
   - WORD directives for SIMD instructions with hex encodings
   - Native syntax for branch instructions
   - Inline comments showing original assembly

## Requirements

- `clang` - C compiler with SIMD support
- `llvm-objdump` or `objdump` - For binary extraction

## Architecture Support

- **ARM64**: Full support (tested and working)
- **AMD64**: Framework ready (needs testing)
