// Package main is a code generator for vecgo SIMD functions.
// Converts C source files to Go assembly using clang + objdump
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const archARM64 = "arm64"

var (
	verbose     = flag.Bool("v", false, "verbose output")
	optimize    = flag.String("O", "3", "optimization level")
	output      = flag.String("o", "", "output directory (default: internal/simd)")
	arch        = flag.String("arch", "", "target architecture (arm64, amd64)")
	goos        = flag.String("goos", "", "target OS (linux, darwin)")
	noTail      = flag.Bool("no-tail", false, "define SIMD_NO_TAIL when compiling C (omit scalar tails in kernels that support it)")
	emitGoStubs = flag.Bool("go-stubs", true, "emit a Go file containing //go:noescape declarations matching parsed C signatures")
	allowRelocs = flag.Bool("allow-relocs", false, "allow relocations in the compiled object (unsafe with raw WORD emission)")
	keepTemp    = flag.Bool("keep-temp", false, "keep temporary assembly/object files for debugging")
	defines     multiFlag
)

func init() {
	flag.Var(&defines, "D", "additional preprocessor define to pass to clang (repeatable)")
}

func main() {
	flag.Parse()

	if *arch == "" || *goos == "" {
		fmt.Fprintln(os.Stderr, "must specify -arch and -goos explicitly")
		os.Exit(1)
	}

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <source.c>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	sourcePath := flag.Arg(0)
	outputDir := *output
	if outputDir == "" {
		outputDir = filepath.Dir(filepath.Dir(sourcePath))
	}

	gen := &Generator{
		SourcePath:  sourcePath,
		OutputDir:   outputDir,
		Verbose:     *verbose,
		OptLevel:    *optimize,
		Arch:        *arch,
		OS:          *goos,
		NoTail:      *noTail,
		EmitGoStubs: *emitGoStubs,
		AllowReloc:  *allowRelocs,
		KeepTemp:    *keepTemp,
		Defines:     append([]string(nil), defines...),
	}

	if err := gen.Generate(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ Generated Go assembly from %s\n", sourcePath)
}

type Generator struct {
	SourcePath string
	OutputDir  string
	Verbose    bool
	OptLevel   string
	Arch       string
	OS         string

	NoTail      bool
	EmitGoStubs bool
	AllowReloc  bool
	KeepTemp    bool
	Defines     []string
}

type multiFlag []string

func (m *multiFlag) String() string { return strings.Join(*m, ",") }

func (m *multiFlag) Set(value string) error {
	*m = append(*m, value)
	return nil
}

type Function struct {
	Name       string
	Lines      []AsmLine
	Parameters []Parameter
	ArgOffsets map[string]int
	ReturnType string
}

type Parameter struct {
	Name string
	Type string
}

type AsmLine struct {
	Labels   []string
	Assembly string
	Binary   []string
}

func (g *Generator) Generate() error {
	asmPath, err := g.compileToAssembly()
	if err != nil {
		return err
	}
	if !g.KeepTemp {
		defer func() { _ = os.Remove(asmPath) }()
	}

	objPath, err := g.compileToObject(asmPath)
	if err != nil {
		return err
	}
	if !g.KeepTemp {
		defer func() { _ = os.Remove(objPath) }()
	}

	if !g.AllowReloc {
		if err := g.validateNoRelocations(objPath); err != nil {
			return err
		}
	}

	// Parse C source to extract function signatures
	functionSigs, err := g.parseCFunctions()
	if err != nil {
		return err
	}

	functions, err := g.parseAssembly(asmPath)
	if err != nil {
		return err
	}

	// Attach parameter info from C source to functions
	for name, fn := range functions {
		if sig, ok := functionSigs[name]; ok {
			fn.Parameters = sig.Parameters
			fn.ReturnType = sig.ReturnType
			fn.ArgOffsets = sig.ArgOffsets
		}
	}

	if err := g.extractBinary(objPath, functions); err != nil {
		return err
	}

	if err := g.generateGoAsm(functions); err != nil {
		return err
	}
	if g.EmitGoStubs {
		if err := g.generateGoStubs(functions); err != nil {
			return err
		}
	}
	return nil
}

// ------------------------------------------------------------
// C Function Parsing
// ------------------------------------------------------------

// FunctionSignature represents a parsed C function signature
type FunctionSignature struct {
	Name       string
	Parameters []Parameter
	ArgOffsets map[string]int
	ReturnType string
}

func (g *Generator) parseCFunctions() (map[string]*FunctionSignature, error) {
	content, err := os.ReadFile(g.SourcePath)
	if err != nil {
		return nil, err
	}

	signatures := make(map[string]*FunctionSignature)
	text := string(content)

	// Match function definitions: return_type function_name(params)
	// Handle return types like "void", "float", "long long", "int64_t", etc.
	funcRe := regexp.MustCompile(`(?m)^((?:unsigned\s+)?(?:long\s+long|long|int|short|char|float|double|void|int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t))\s+(\w+)\s*\(([^)]*)\)`)

	matches := funcRe.FindAllStringSubmatch(text, -1)
	for _, m := range matches {
		retType := strings.TrimSpace(m[1])
		funcName := m[2]
		paramsStr := m[3]

		sig := &FunctionSignature{
			Name:       funcName,
			ReturnType: retType,
			ArgOffsets: make(map[string]int),
		}

		if paramsStr != "" && paramsStr != "void" { //nolint:goconst // inline string is clearer in code generator
			params := strings.Split(paramsStr, ",")
			offset := 0
			for _, p := range params {
				p = strings.TrimSpace(p)
				// Remove __restrict__ and const modifiers
				p = strings.ReplaceAll(p, "__restrict__", "")
				p = strings.ReplaceAll(p, "const ", "")
				p = strings.TrimSpace(p)

				// Check if this is a pointer type (has * anywhere)
				isPointer := strings.Contains(p, "*")
				// Remove all * for easier parsing
				p = strings.ReplaceAll(p, "*", " ")
				p = strings.Join(strings.Fields(p), " ") // normalize whitespace

				// Extract type and name
				parts := strings.Fields(p)
				if len(parts) >= 2 {
					paramName := parts[len(parts)-1]
					paramType := strings.Join(parts[:len(parts)-1], " ")
					// Add back pointer indicator if this was a pointer type
					if isPointer {
						paramType += " *"
					}

					sig.Parameters = append(sig.Parameters, Parameter{
						Name: paramName,
						Type: paramType,
					})
					sig.ArgOffsets[paramName] = offset
					offset += 8 // 64-bit platform
				}
			}
		}

		signatures[funcName] = sig
	}

	return signatures, nil
}

func (g *Generator) generateGoStubs(functions map[string]*Function) error {
	if !g.EmitGoStubs {
		return nil
	}

	base := strings.TrimSuffix(filepath.Base(g.SourcePath), filepath.Ext(g.SourcePath))
	// Use base_stubs.go (not base_stubs_arch.go) since build constraint already specifies arch
	goFile := filepath.Join(g.OutputDir, base+"_stubs.go")

	var buf bytes.Buffer
	buf.WriteString("// Code generated by internal/simd/cmd/generator. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "//go:build !noasm && %s\n\n", g.Arch)
	buf.WriteString("package simd\n\n")

	// Check if we need unsafe import (for pointer parameters)
	needsUnsafe := false
	for _, fn := range functions {
		for _, p := range fn.Parameters {
			if isPointerType(p.Type) {
				needsUnsafe = true
				break
			}
		}
		if needsUnsafe {
			break
		}
	}
	if needsUnsafe {
		buf.WriteString("import \"unsafe\"\n\n")
	}

	names := make([]string, 0, len(functions))
	for name := range functions {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		fn := functions[name]
		buf.WriteString("//go:noescape\n")

		// Build parameter list
		var params []string
		for _, p := range fn.Parameters {
			goType := cTypeToGo(p.Type)
			params = append(params, p.Name+" "+goType)
		}

		retType := ""
		if fn.ReturnType != "void" && fn.ReturnType != "" {
			retType = " " + cTypeToGo(fn.ReturnType)
		}

		fmt.Fprintf(&buf, "func %s(%s)%s\n\n", name, strings.Join(params, ", "), retType)
	}

	return os.WriteFile(goFile, buf.Bytes(), 0600)
}

// isPointerType checks if a C type is a pointer type.
func isPointerType(cType string) bool {
	return strings.Contains(cType, "*")
}

// isFloatType checks if a C type is a floating point type.
func isFloatType(cType string) bool {
	cType = strings.TrimSpace(cType)
	// Check for pointer types - pointers are not floats
	if strings.Contains(cType, "*") {
		return false
	}
	return cType == "float" || cType == "double" ||
		cType == "float32" || cType == "float64"
}

func cTypeToGo(cType string) string {
	cType = strings.TrimSpace(cType)

	// Check for pointer types first (before stripping *)
	if strings.Contains(cType, "*") {
		return "unsafe.Pointer"
	}

	switch cType {
	case "float":
		return "float32"
	case "double":
		return "float64"
	case "int", "int32_t":
		return "int32"
	case "long", "int64_t", "long long":
		return "int64"
	case "uint8_t", "unsigned char":
		return "uint8"
	case "uint16_t", "unsigned short":
		return "uint16"
	case "uint32_t", "unsigned int":
		return "uint32"
	case "uint64_t", "unsigned long", "unsigned long long":
		return "uint64"
	case "int8_t", "char":
		return "int8"
	case "int16_t", "short":
		return "int16"
	case "void":
		return ""
	default:
		return "uintptr" // Default for unknown types
	}
}

// ------------------------------------------------------------
// Clang helpers
// ------------------------------------------------------------

func (g *Generator) getClangTarget() string {
	archMap := map[string]string{
		"amd64":   "x86_64",
		archARM64: archARM64,
	}
	osMap := map[string]string{
		"linux":  "linux-gnu",
		"darwin": "apple-darwin",
	}
	return fmt.Sprintf("%s-%s", archMap[g.Arch], osMap[g.OS])
}

func (g *Generator) compileToAssembly() (string, error) {
	base := strings.TrimSuffix(filepath.Base(g.SourcePath), filepath.Ext(g.SourcePath))
	tmp, err := os.CreateTemp("", base+"_*.s")
	if err != nil {
		return "", err
	}
	asmPath := tmp.Name()
	_ = tmp.Close()

	args := []string{
		"-S",
		"-O" + g.OptLevel,
		"-target", g.getClangTarget(),
		"-fno-asynchronous-unwind-tables",
		"-fno-exceptions",
		"-fno-rtti",
		// The generator emits raw WORD/BYTE for the text section only.
		// Disable clang vectorization passes that may introduce literal pools
		// (and thus relocations) for shuffle masks and other constants.
		"-fno-vectorize",
		"-fno-slp-vectorize",
	}

	if g.NoTail {
		args = append(args, "-DSIMD_NO_TAIL")
	}
	for _, d := range g.Defines {
		args = append(args, "-D"+d)
	}

	if g.Arch == archARM64 {
		args = append(args, "-ffixed-x18")

		// Detect SIMD instruction set from filename and add appropriate flags
		baseName := strings.ToLower(filepath.Base(g.SourcePath))
		if strings.Contains(baseName, "sve2") {
			args = append(args, "-march=armv9-a+sve2", "-D__ARM_FEATURE_SVE2=1")
		}
	}
	if g.Arch == "amd64" {
		// -mno-red-zone: required for Go compatibility (Go doesn't use red zone)
		// -fomit-frame-pointer: CRITICAL - prevent push rbp/mov rbp,rsp prologue
		//                       which breaks Go's frame pointer-based argument access
		// Note: Do NOT use -mstackrealign - it forces frame pointer setup despite -fomit-frame-pointer
		args = append(args, "-mno-red-zone", "-fomit-frame-pointer")

		// Detect SIMD instruction set from filename and add appropriate flags
		baseName := strings.ToLower(filepath.Base(g.SourcePath))
		if strings.Contains(baseName, "avx512") {
			args = append(args, "-mavx512f", "-mavx512dq", "-mavx2", "-mfma", "-mf16c", "-mavx512vl", "-mavx512bw", "-mavx512vpopcntdq")
		} else if strings.Contains(baseName, "avx") {
			args = append(args, "-mavx2", "-mfma", "-mf16c")
		}
	}

	args = append(args, g.SourcePath, "-o", asmPath)
	return asmPath, g.runCommand("clang", args...)
}

func (g *Generator) compileToObject(asmPath string) (string, error) {
	objPath := strings.TrimSuffix(asmPath, ".s") + ".o"
	args := []string{
		"-target", g.getClangTarget(),
		"-c", asmPath,
		"-o", objPath,
	}

	// For SVE2 files, we need the same march flag for assembly
	if g.Arch == archARM64 {
		baseName := strings.ToLower(filepath.Base(g.SourcePath))
		if strings.Contains(baseName, "sve2") {
			args = append(args, "-march=armv9-a+sve2")
		}
	}

	return objPath, g.runCommand("clang", args...)
}

func (g *Generator) validateNoRelocations(objPath string) error {
	// Raw WORD/BYTE emission cannot carry relocations, so any relocation in the
	// compiled object means the generated Go assembly may be incorrect.
	//
	// Note: llvm-objdump prints relocations only when present.
	cmd := exec.Command("llvm-objdump", "-r", objPath)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("llvm-objdump -r failed: %w\n%s", err, strings.TrimSpace(out.String()))
	}
	text := strings.TrimSpace(out.String())
	if text == "" {
		return nil
	}

	// Heuristic: relocation dumps contain 'RELOCATION RECORDS' headers.
	if strings.Contains(text, "RELOCATION") || strings.Contains(text, "Relocations") {
		return fmt.Errorf("object contains relocations; raw WORD emission is unsafe. Rework the C to avoid literal pools/calls, or rerun with -allow-relocs.\nllvm-objdump -r output:\n%s", text)
	}
	return nil
}

// ------------------------------------------------------------
// Assembly parsing
// ------------------------------------------------------------

func (g *Generator) parseAssembly(asmPath string) (map[string]*Function, error) {
	file, err := os.Open(asmPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	nameLine := regexp.MustCompile(`^(\w+):`)
	labelLine := regexp.MustCompile(`^\.(\w+):`)
	codeLine := regexp.MustCompile(`^\s+[\w\.]`)

	functions := make(map[string]*Function)
	var current string

	sc := bufio.NewScanner(file)
	for sc.Scan() {
		line := sc.Text()
		trim := strings.TrimSpace(line)

		if trim == "" || strings.HasPrefix(trim, "#") {
			continue
		}
		// Skip directives
		if strings.HasPrefix(trim, ".") && !labelLine.MatchString(line) {
			continue
		}

		if m := nameLine.FindStringSubmatch(line); m != nil {
			current = m[1]
			functions[current] = &Function{Name: current}
			continue
		}

		if current == "" {
			continue
		}

		if m := labelLine.FindStringSubmatch(line); m != nil {
			functions[current].Lines = append(functions[current].Lines, AsmLine{
				Labels: []string{m[1]},
			})
			continue
		}

		if codeLine.MatchString(line) {
			asm := strings.TrimSpace(line)
			if i := strings.IndexAny(asm, "#/"); i >= 0 {
				asm = strings.TrimSpace(asm[:i])
			}
			if asm != "" {
				functions[current].Lines = append(functions[current].Lines, AsmLine{
					Assembly: asm,
				})
			}
		}
	}

	return functions, sc.Err()
}

// ------------------------------------------------------------
// Binary extraction
// ------------------------------------------------------------

func (g *Generator) extractBinary(objPath string, functions map[string]*Function) error {
	var out bytes.Buffer

	cmd := exec.Command("llvm-objdump", "-d", objPath)
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return err
	}

	sym := regexp.MustCompile(`^[0-9a-f]+ <(\w+)>:`)

	var fn *Function
	idx := 0
	var lastAddr int64 = -1

	sc := bufio.NewScanner(&out)
	for sc.Scan() {
		line := sc.Text()
		if g.Verbose {
			fmt.Printf("Scanning line: %q\n", line)
		}

		if m := sym.FindStringSubmatch(line); m != nil {
			fn = functions[m[1]]
			idx = 0
			lastAddr = -1
			continue
		}

		if fn == nil {
			continue
		}

		// Parse instruction lines: "   offset: bytes..."
		if (strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t")) && strings.Contains(line, ":") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) != 2 {
				continue
			}

			// Parse address to check for gaps (padding)
			addrStr := strings.TrimSpace(parts[0])
			addr, err := strconv.ParseInt(addrStr, 16, 64)
			if err == nil {
				if lastAddr != -1 && addr > lastAddr {
					gap := addr - lastAddr
					if gap > 0 {
						// Detected padding gap
						// Insert NOPs into the Go assembly structure to maintain alignment
						// ARM64 NOP is 4 bytes (0xd503201f)
						// x86 NOP is 1 byte (0x90)
						var nopBytes []string
						var stride int64

						if g.Arch == "arm64" {
							nopBytes = []string{"d503201f"}
							stride = 4
						} else {
							nopBytes = []string{"90"}
							stride = 1
						}

						if gap%stride == 0 {
							count := int(gap / stride)
							nopLine := AsmLine{
								Assembly: "// NOP padding",
								Binary:   nopBytes,
							}

							if g.Verbose {
								fmt.Printf("Inserting %d NOPs at idx %d (gap %d)\n", count, idx, gap)
							}

							for i := 0; i < count; i++ {
								// Find generic insertion point (skip empty/labels)
								// Actually, strict insertion at idx is better to keep sync with binary flow
								// But idx points to next *source* line.
								// If we skip labels in idx search, we should respect that.
								// But here we are inserting *before* the current instruction match.

								// We insert at the current consumption cursor 'idx'.
								// Note: We normally advance idx to skip labels *before* assigning.
								// But here we interact with the stream.

								// Insert into slice
								if idx >= len(fn.Lines) {
									fn.Lines = append(fn.Lines, nopLine)
								} else {
									fn.Lines = append(fn.Lines[:idx], append([]AsmLine{nopLine}, fn.Lines[idx:]...)...)
								}
								idx++
							}
							// Update lastAddr so we don't re-detect this gap or count it wrong
							lastAddr += gap
						}
					}
				}
			}

			content := parts[1]

			// If there is a tab, the bytes are before the tab
			if tabIdx := strings.Index(content, "\t"); tabIdx != -1 {
				content = content[:tabIdx]
			}

			// Parse hex bytes or words
			fields := strings.Fields(content)
			var bytes []string
			for _, p := range fields {
				if (len(p) == 2 || len(p) == 8) && isHex(p) {
					bytes = append(bytes, p)
				}
			}

			if len(bytes) > 0 {
				if g.Verbose {
					fmt.Printf("Matched binary: %v for line idx %d\n", bytes, idx)
				}

				// Update address tracker
				size := 0
				for _, b := range bytes {
					if len(b) == 2 {
						size++
					} else if len(b) == 8 {
						size += 4
					}
				}
				// If we failed parsing address earlier, assume continuity?
				// But we handle gaps primarily.
				// If parsing worked, use it. If not, best effort.
				if err == nil {
					lastAddr = addr + int64(size)
				}

				for idx < len(fn.Lines) && (len(fn.Lines[idx].Binary) > 0 || fn.Lines[idx].Assembly == "") {
					idx++
				}
				if idx < len(fn.Lines) {
					fn.Lines[idx].Binary = bytes
					idx++
				}
			}
		}
	}
	return sc.Err()
}

func isHex(s string) bool {
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') && (c < 'A' || c > 'F') {
			return false
		}
	}
	return true
}

// ------------------------------------------------------------
// Go assembly generation
// ------------------------------------------------------------

func (g *Generator) generateGoAsm(functions map[string]*Function) error {
	base := strings.TrimSuffix(filepath.Base(g.SourcePath), filepath.Ext(g.SourcePath))
	out := filepath.Join(g.OutputDir, base+".s")

	var buf bytes.Buffer
	buf.WriteString("// Code generated by internal/simd/cmd/generator. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "//go:build !noasm && %s\n\n", g.Arch)
	buf.WriteString("#include \"textflag.h\"\n\n")

	names := make([]string, 0, len(functions))
	for name := range functions {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		fn := functions[name]
		// Calculate frame size: each parameter is 8 bytes on 64-bit platforms
		numParams := len(fn.Parameters)
		if numParams == 0 {
			// Default to 4 parameters if not parsed
			numParams = 4
		}
		argSize := numParams * 8
		retSize := 0
		if fn.ReturnType != "void" && fn.ReturnType != "" {
			// Go ABI uses a caller-allocated return slot in the args area.
			// Most of our non-void kernels return int64/long long.
			retSize = 8
		}
		frameSize := argSize + retSize

		fmt.Fprintf(&buf, "TEXT ·%s(SB), NOSPLIT, $0-%d\n", fn.Name, frameSize)

		switch g.Arch {
		case "arm64":
			// Generate parameter loads based on actual parameters
			// ARM64 ABI: integers/pointers in R0-R7, floats/doubles in D0-D7
			intRegisters := []string{"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}
			floatRegisters := []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"}
			intRegIdx := 0
			floatRegIdx := 0
			for i, param := range fn.Parameters {
				offset := i * 8
				if customOffset, ok := fn.ArgOffsets[param.Name]; ok {
					offset = customOffset
				}
				if isFloatType(param.Type) {
					if floatRegIdx < len(floatRegisters) {
						fmt.Fprintf(&buf, "\tFMOVD %s+%d(FP), %s\n", param.Name, offset, floatRegisters[floatRegIdx])
						floatRegIdx++
					}
				} else {
					if intRegIdx < len(intRegisters) {
						fmt.Fprintf(&buf, "\tMOVD %s+%d(FP), %s\n", param.Name, offset, intRegisters[intRegIdx])
						intRegIdx++
					}
				}
			}
			buf.WriteString("\n")
		case "amd64":
			// Generate parameter loads for AMD64 (System V ABI)
			// Integers/pointers go to DI, SI, DX, CX, R8, R9
			// Floats/doubles go to XMM0-XMM7
			intRegisters := []string{"DI", "SI", "DX", "CX", "R8", "R9"}
			floatRegisters := []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"}
			intRegIdx := 0
			floatRegIdx := 0
			for i, param := range fn.Parameters {
				offset := i * 8
				if customOffset, ok := fn.ArgOffsets[param.Name]; ok {
					offset = customOffset
				}
				if isFloatType(param.Type) {
					if floatRegIdx < len(floatRegisters) {
						fmt.Fprintf(&buf, "\tMOVSD %s+%d(FP), %s\n", param.Name, offset, floatRegisters[floatRegIdx])
						floatRegIdx++
					}
				} else {
					if intRegIdx < len(intRegisters) {
						fmt.Fprintf(&buf, "\tMOVQ %s+%d(FP), %s\n", param.Name, offset, intRegisters[intRegIdx])
						intRegIdx++
					}
				}
			}
			buf.WriteString("\n")
		}

		for _, l := range fn.Lines {
			for _, lbl := range l.Labels {
				fmt.Fprintf(&buf, "%s:\n", lbl)
			}

			if l.Assembly == "ret" || l.Assembly == "retq" {
				if fn.ReturnType != "void" && fn.ReturnType != "" {
					retOffset := argSize
					if g.Arch == "arm64" {
						if strings.Contains(fn.ReturnType, "float") {
							fmt.Fprintf(&buf, "\tFMOVS S0, ret+%d(FP)\n", retOffset)
						} else {
							fmt.Fprintf(&buf, "\tMOVD R0, ret+%d(FP)\n", retOffset)
						}
					} else if g.Arch == "amd64" {
						if strings.Contains(fn.ReturnType, "float") {
							fmt.Fprintf(&buf, "\tMOVSS XMM0, ret+%d(FP)\n", retOffset)
						} else {
							fmt.Fprintf(&buf, "\tMOVQ AX, ret+%d(FP)\n", retOffset)
						}
					}
				}
				buf.WriteString("\tRET\n")
				continue
			}

			if len(l.Binary) > 0 {
				for _, b := range l.Binary {
					if len(b) == 8 {
						fmt.Fprintf(&buf, "\tWORD $0x%s\n", b)
					} else {
						fmt.Fprintf(&buf, "\tBYTE $0x%s\n", b)
					}
				}
			}
		}
		buf.WriteString("\n")
	}

	return os.WriteFile(out, buf.Bytes(), 0600)
}

// ------------------------------------------------------------
// Utils
// ------------------------------------------------------------

func (g *Generator) runCommand(name string, args ...string) error {
	if g.Verbose {
		fmt.Println(name, strings.Join(args, " "))
	}
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
