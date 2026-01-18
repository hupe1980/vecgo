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
	"strings"
)

const (
	archARM64 = "arm64"
	archAMD64 = "amd64"
)

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
	Disasm   string // Original disassembly from objdump (for comments)
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
		archAMD64: "x86_64",
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
	if g.Arch == archAMD64 {
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
// AMD64 Jump Instruction Handling
// ------------------------------------------------------------

// amd64JumpOpcodes maps x86 jump mnemonics to Go assembly mnemonics
var amd64JumpOpcodes = map[string]string{
	"jmp":  "JMP",
	"je":   "JE",
	"jz":   "JE", // alias
	"jne":  "JNE",
	"jnz":  "JNE", // alias
	"jl":   "JLT",
	"jlt":  "JLT",
	"jle":  "JLE",
	"jg":   "JGT",
	"jgt":  "JGT",
	"jge":  "JGE",
	"jb":   "JCS", // below (unsigned) = carry set
	"jnae": "JCS", // alias
	"jc":   "JCS", // carry
	"jae":  "JCC", // above or equal (unsigned) = carry clear
	"jnb":  "JCC", // alias
	"jnc":  "JCC", // no carry
	"ja":   "JHI", // above (unsigned)
	"jnbe": "JHI", // alias
	"jbe":  "JLS", // below or equal (unsigned)
	"jna":  "JLS", // alias
	"js":   "JMI", // sign (negative)
	"jns":  "JPL", // not sign (positive)
	"jo":   "JOS", // overflow
	"jno":  "JOC", // no overflow
	"jp":   "JPE", // parity even
	"jpe":  "JPE", // alias
	"jnp":  "JPO", // parity odd
	"jpo":  "JPO", // alias
}

// isAMD64Jump returns true if the assembly represents a jump instruction.
// We check the original assembly source (e.g., "jge .LBB0_2") not objdump output.
func isAMD64Jump(asm string) bool {
	// Assembly format from source: "jmp .LBB0_8" or "jge .LBB0_2"
	// Check if it starts with "j" (all x86 jumps do: jmp, je, jne, jge, etc.)
	return strings.HasPrefix(asm, "j")
}

// convertAMD64JumpToGo converts an AMD64 jump from assembly source to Go assembly.
// Input format (from clang -S): "jmp .LBB0_8" or "jge .LBB0_2"
// Output format: "JMP LBB0_8" or "JGE LBB0_2"
// Following goat's approach: https://github.com/gorse-io/goat
func convertAMD64JumpToGo(asm string) string {
	// Split on the label (format: "jmp .LBB0_8" or "jge .LBB0_2")
	// The label is separated by space or tab
	parts := strings.Fields(asm)
	if len(parts) < 2 {
		return ""
	}

	opcode := parts[0]
	label := parts[1]

	// Get the Go opcode (uppercase)
	goOp, ok := amd64JumpOpcodes[opcode]
	if !ok {
		// Unknown jump - use uppercase opcode directly
		goOp = strings.ToUpper(opcode)
	}

	// Handle label: ".LBB0_8" -> "LBB0_8"
	// Strip the leading dot (Go asm labels don't have dots)
	label = strings.TrimPrefix(label, ".")

	return fmt.Sprintf("%s %s", goOp, label)
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
	var pendingLabels []string // Track labels that need to be merged with next code line

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
			pendingLabels = nil // Reset pending labels for new function
			continue
		}

		if current == "" {
			continue
		}

		if m := labelLine.FindStringSubmatch(line); m != nil {
			// Accumulate labels - they will be attached to the next code line
			pendingLabels = append(pendingLabels, m[1])
			continue
		}

		if codeLine.MatchString(line) {
			asm := strings.TrimSpace(line)
			if i := strings.IndexAny(asm, "#/"); i >= 0 {
				asm = strings.TrimSpace(asm[:i])
			}
			if asm != "" {
				// Create line with any pending labels merged in
				functions[current].Lines = append(functions[current].Lines, AsmLine{
					Labels:   pendingLabels,
					Assembly: asm,
				})
				pendingLabels = nil // Clear pending labels after attaching
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

	sc := bufio.NewScanner(&out)
	for sc.Scan() {
		line := sc.Text()
		if g.Verbose {
			fmt.Printf("Scanning line: %q\n", line)
		}

		if m := sym.FindStringSubmatch(line); m != nil {
			fn = functions[m[1]]
			idx = 0
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

			content := parts[1]

			// Extract the disassembly mnemonic (after the tab)
			var mnemonic string
			if tabIdx := strings.Index(content, "\t"); tabIdx != -1 {
				mnemonic = strings.TrimSpace(content[tabIdx+1:])
				content = content[:tabIdx]
			}

			// Skip NOP instructions - they are padding inserted by the compiler
			// and don't have corresponding lines in the assembly source
			if strings.Contains(strings.ToLower(mnemonic), "nop") {
				continue
			}

			// Parse hex bytes or words
			fields := strings.Fields(content)
			var bytes []string
			for _, p := range fields {
				if len(p) == 2 && isHex(p) {
					// x86: individual bytes like "c5 f8 77"
					bytes = append(bytes, p)
				} else if len(p) == 8 && isHex(p) {
					// ARM64: 4-byte words like "f100205f"
					// Split into individual bytes (little-endian order for ARM)
					bytes = append(bytes, p[6:8], p[4:6], p[2:4], p[0:2])
				}
			}

			if len(bytes) > 0 {
				if g.Verbose {
					fmt.Printf("Matched binary: %v for line idx %d\n", bytes, idx)
				}

				for idx < len(fn.Lines) && (len(fn.Lines[idx].Binary) > 0 || fn.Lines[idx].Assembly == "") {
					idx++
				}
				if idx < len(fn.Lines) {
					fn.Lines[idx].Binary = bytes
					fn.Lines[idx].Disasm = mnemonic // Store disassembly for comments
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
		case archAMD64:
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
					} else if g.Arch == archAMD64 {
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

			// For AMD64, convert jump instructions to native Go assembly mnemonics.
			// This is critical because raw bytes contain PC-relative offsets that
			// were computed for the original object file, but Go's assembler needs
			// symbolic labels to compute correct offsets during linking.
			// We use l.Assembly (from source .s file) which has labels like ".LBB0_2"
			// rather than l.Disasm (from objdump) which has function-relative offsets.
			if g.Arch == archAMD64 && l.Assembly != "" && isAMD64Jump(l.Assembly) {
				goJump := convertAMD64JumpToGo(l.Assembly)
				if goJump != "" {
					fmt.Fprintf(&buf, "\t%s\n", goJump)
					continue
				}
			}

			if len(l.Binary) > 0 {
				// Pack bytes into QUAD/LONG/WORD for more compact output
				// Note: ARM64 uses WORD for 4-byte values (not LONG)
				buf.WriteString("\t")
				pos := 0
				for pos < len(l.Binary) {
					if pos > 0 {
						buf.WriteString("; ")
					}
					remaining := len(l.Binary) - pos
					if remaining >= 8 && g.Arch == archAMD64 {
						// QUAD: 8 bytes (little-endian) - AMD64 only
						fmt.Fprintf(&buf, "QUAD $0x%s%s%s%s%s%s%s%s",
							l.Binary[pos+7], l.Binary[pos+6], l.Binary[pos+5], l.Binary[pos+4],
							l.Binary[pos+3], l.Binary[pos+2], l.Binary[pos+1], l.Binary[pos])
						pos += 8
					} else if remaining >= 4 && g.Arch == archAMD64 {
						// LONG: 4 bytes (little-endian) - AMD64
						fmt.Fprintf(&buf, "LONG $0x%s%s%s%s",
							l.Binary[pos+3], l.Binary[pos+2], l.Binary[pos+1], l.Binary[pos])
						pos += 4
					} else if remaining >= 4 && g.Arch == "arm64" {
						// WORD: 4 bytes for ARM64 (ARM uses WORD for 32-bit)
						fmt.Fprintf(&buf, "WORD $0x%s%s%s%s",
							l.Binary[pos+3], l.Binary[pos+2], l.Binary[pos+1], l.Binary[pos])
						pos += 4
					} else if remaining >= 2 {
						// WORD: 2 bytes (little-endian) - AMD64 only
						fmt.Fprintf(&buf, "WORD $0x%s%s", l.Binary[pos+1], l.Binary[pos])
						pos += 2
					} else {
						// BYTE: 1 byte
						fmt.Fprintf(&buf, "BYTE $0x%s", l.Binary[pos])
						pos++
					}
				}
				// Add disassembly as comment
				if l.Disasm != "" {
					fmt.Fprintf(&buf, " // %s", l.Disasm)
				}
				buf.WriteString("\n")
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
