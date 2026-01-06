package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var goIdentRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

func (g *Generator) generateGoStubs(functions map[string]*Function) error {
	base := strings.TrimSuffix(filepath.Base(g.SourcePath), filepath.Ext(g.SourcePath))
	out := filepath.Join(g.OutputDir, base+"_stubs.go")

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "//go:build !noasm && %s\n\n", g.Arch)
	buf.WriteString("package simd\n\n")
	buf.WriteString("import \"unsafe\"\n\n")

	names := make([]string, 0, len(functions))
	for name := range functions {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		fn := functions[name]
		if len(fn.Parameters) == 0 {
			// Skip helper symbols or functions without parsed signatures.
			continue
		}

		params := make([]string, 0, len(fn.Parameters))
		used := make(map[string]int, len(fn.Parameters))
		for i, p := range fn.Parameters {
			paramName := sanitizeGoIdent(p.Name)
			if paramName == "" {
				paramName = fmt.Sprintf("arg%d", i)
			}
			if n := used[paramName]; n > 0 {
				paramName = fmt.Sprintf("%s%d", paramName, n+1)
			}
			used[paramName]++

			params = append(params, fmt.Sprintf("%s %s", paramName, goTypeFromC(p.Type)))
		}

		buf.WriteString("//go:noescape\n")
		fmt.Fprintf(&buf, "func %s(%s)%s\n\n", fn.Name, strings.Join(params, ", "), goReturnTypeFromC(fn.ReturnType))
	}

	return os.WriteFile(out, buf.Bytes(), 0600)
}

func sanitizeGoIdent(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}
	name = strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= 'A' && r <= 'Z':
			return r
		case r >= '0' && r <= '9':
			return r
		case r == '_':
			return r
		default:
			return '_'
		}
	}, name)
	if !goIdentRe.MatchString(name) {
		return ""
	}
	if isGoKeyword(name) {
		return "_" + name
	}
	return name
}

func isGoKeyword(s string) bool {
	switch s {
	case "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func", "go", "goto", "if", "import", "interface", "map", "package", "range", "return", "select", "struct", "switch", "type", "var":
		return true
	default:
		return false
	}
}

func goTypeFromC(cType string) string {
	t := strings.TrimSpace(cType)
	if t == "" {
		return "unsafe.Pointer"
	}
	if strings.HasSuffix(t, "*") {
		return "unsafe.Pointer"
	}

	lower := strings.ToLower(t)
	switch {
	case strings.Contains(lower, "uint64"):
		return "uint64"
	case strings.Contains(lower, "int64"):
		return "int64"
	case strings.Contains(lower, "float"):
		// Avoid float32 parameters; our kernels should take float args by pointer.
		return "unsafe.Pointer"
	default:
		// Keep all scalar integers as 64-bit to match our generator's 8-byte slot model.
		return "int64"
	}
}

func goReturnTypeFromC(cRet string) string {
	rt := strings.TrimSpace(cRet)
	if rt == "" || rt == "void" {
		return ""
	}
	if strings.Contains(strings.ToLower(rt), "float") {
		return " float32"
	}
	// Most SIMD kernels return long long / int64.
	return " int64"
}
