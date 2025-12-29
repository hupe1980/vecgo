// Code generator for vecgo SIMD functions
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
	"strings"
)

var (
	verbose  = flag.Bool("v", false, "verbose output")
	optimize = flag.String("O", "3", "optimization level")
	output   = flag.String("o", "", "output directory (default: same as source)")
	pkg      = flag.String("pkg", "math32", "package name")
	arch     = flag.String("arch", "", "target architecture (arm64, amd64)")
	goos     = flag.String("goos", "", "target OS (linux, darwin)")
)

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
		SourcePath: sourcePath,
		OutputDir:  outputDir,
		Package:    *pkg,
		Verbose:    *verbose,
		OptLevel:   *optimize,
		Arch:       *arch,
		OS:         *goos,
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
	Package    string
	Verbose    bool
	OptLevel   string
	Arch       string
	OS         string
}

type Function struct {
	Name       string
	Lines      []AsmLine
	Parameters []Parameter
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
	defer os.Remove(asmPath)

	objPath, err := g.compileToObject(asmPath)
	if err != nil {
		return err
	}
	defer os.Remove(objPath)

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
		}
	}

	if err := g.extractBinary(objPath, functions); err != nil {
		return err
	}

	return g.generateGoAsm(functions)
}

// ------------------------------------------------------------
// Clang helpers
// ------------------------------------------------------------

func (g *Generator) getClangTarget() string {
	archMap := map[string]string{
		"amd64": "x86_64",
		"arm64": "arm64",
	}
	osMap := map[string]string{
		"linux":  "linux-gnu",
		"darwin": "apple-darwin",
	}
	return fmt.Sprintf("%s-%s", archMap[g.Arch], osMap[g.OS])
}

func (g *Generator) compileToAssembly() (string, error) {
	base := strings.TrimSuffix(filepath.Base(g.SourcePath), filepath.Ext(g.SourcePath))
	asmPath := filepath.Join(os.TempDir(), base+"_tmp.s")

	args := []string{
		"-S",
		"-O" + g.OptLevel,
		"-target", g.getClangTarget(),
		"-fno-asynchronous-unwind-tables",
		"-fno-exceptions",
		"-fno-rtti",
	}

	if g.Arch == "arm64" {
		args = append(args, "-ffixed-x18")
	}
	if g.Arch == "amd64" {
		args = append(args, "-mno-red-zone", "-mstackrealign")

		// Detect SIMD instruction set from filename and add appropriate flags
		baseName := strings.ToLower(filepath.Base(g.SourcePath))
		if strings.Contains(baseName, "avx512") {
			args = append(args, "-mavx512f", "-mavx512dq", "-mavx2", "-mfma")
		} else if strings.Contains(baseName, "avx") {
			args = append(args, "-mavx2", "-mfma")
		}
	}

	if g.Arch == "arm64" {
		// ARM NEON is enabled by default on ARM64
	}

	args = append(args, g.SourcePath, "-o", asmPath)
	return asmPath, g.runCommand("clang", args...)
}

func (g *Generator) compileToObject(asmPath string) (string, error) {
	objPath := strings.TrimSuffix(asmPath, ".s") + ".o"
	return objPath, g.runCommand(
		"clang",
		"-target", g.getClangTarget(),
		"-c", asmPath,
		"-o", objPath,
	)
}

// ------------------------------------------------------------
// Assembly parsing
// ------------------------------------------------------------

// parseCFunctions extracts function signatures from C source
func (g *Generator) parseCFunctions() (map[string]*Function, error) {
	data, err := os.ReadFile(g.SourcePath)
	if err != nil {
		return nil, err
	}

	// Match function signatures like: void func_name(type1 *param1, type2 param2, ...)
	funcRe := regexp.MustCompile(`(?m)^void\s+(_\w+)\s*\(([^)]*)\)`)
	functions := make(map[string]*Function)

	matches := funcRe.FindAllStringSubmatch(string(data), -1)
	for _, m := range matches {
		name := m[1]
		paramsStr := m[2]

		fn := &Function{
			Name:       name,
			Parameters: []Parameter{},
		}

		// Parse parameters
		if strings.TrimSpace(paramsStr) != "" {
			paramParts := strings.Split(paramsStr, ",")
			for _, part := range paramParts {
				part = strings.TrimSpace(part)
				// Parse "type *name" or "type name"
				tokens := strings.Fields(part)
				if len(tokens) >= 2 {
					paramType := tokens[0]
					paramName := tokens[len(tokens)-1]
					// Remove * from pointer names
					paramName = strings.TrimPrefix(paramName, "*")
					fn.Parameters = append(fn.Parameters, Parameter{
						Name: paramName,
						Type: paramType,
					})
				}
			}
		}

		functions[name] = fn
	}

	return functions, nil
}

func (g *Generator) parseAssembly(asmPath string) (map[string]*Function, error) {
	file, err := os.Open(asmPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	nameLine := regexp.MustCompile(`^(\w+):`)
	labelLine := regexp.MustCompile(`^\.(\w+):`)
	codeLine := regexp.MustCompile(`^\s+\w+`)

	functions := make(map[string]*Function)
	var current string

	sc := bufio.NewScanner(file)
	for sc.Scan() {
		line := sc.Text()
		trim := strings.TrimSpace(line)

		if trim == "" || strings.HasPrefix(trim, "#") {
			continue
		}
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
		// We look for lines starting with whitespace and containing a colon
		if (strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t")) && strings.Contains(line, ":") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
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
					for idx < len(fn.Lines) && (len(fn.Lines[idx].Binary) > 0 || fn.Lines[idx].Assembly == "") {
						idx++
					}
					if idx < len(fn.Lines) {
						fn.Lines[idx].Binary = bytes
						idx++
					}
				}
			}
		} else {
			if g.Verbose {
				fmt.Printf("No match for line: %s\n", line)
			}
		}
	}
	return sc.Err()
}

func isHex(s string) bool {
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
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
	fmt.Fprintf(&buf, "//go:build !noasm && %s\n\n", g.Arch)
	buf.WriteString("#include \"textflag.h\"\n\n")

	for _, fn := range functions {
		// Calculate frame size: each parameter is 8 bytes on 64-bit platforms
		numParams := len(fn.Parameters)
		if numParams == 0 {
			// Default to 4 parameters if not parsed
			numParams = 4
		}
		frameSize := numParams * 8

		fmt.Fprintf(&buf, "TEXT ·%s(SB), NOSPLIT, $0-%d\n", fn.Name, frameSize)

		switch g.Arch {
		case "arm64":
			// Generate parameter loads based on actual parameters
			registers := []string{"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}
			for i, param := range fn.Parameters {
				if i < len(registers) {
					offset := i * 8
					fmt.Fprintf(&buf, "\tMOVD %s+%d(FP), %s\n", param.Name, offset, registers[i])
				}
			}
			buf.WriteString("\n")
		case "amd64":
			// Generate parameter loads for AMD64 (System V ABI)
			// Pointers and integers go to DI, SI, DX, CX, R8, R9
			registers := []string{"DI", "SI", "DX", "CX", "R8", "R9"}
			for i, param := range fn.Parameters {
				if i < len(registers) {
					offset := i * 8
					fmt.Fprintf(&buf, "\tMOVQ %s+%d(FP), %s\n", param.Name, offset, registers[i])
				}
			}
			buf.WriteString("\n")
		}

		for _, l := range fn.Lines {
			for _, lbl := range l.Labels {
				fmt.Fprintf(&buf, "%s:\n", lbl)
			}

			if l.Assembly == "ret" {
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

	return os.WriteFile(out, buf.Bytes(), 0644)
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
