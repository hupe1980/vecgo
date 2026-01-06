package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"

	"modernc.org/cc/v4"
)

// parseCFunctions extracts function definitions from a C source file using a real C parser.
//
// This is intentionally conservative: we primarily need function names, parameter names/order,
// and whether the return type is void (for Go ABI return slot sizing).
func (g *Generator) parseCFunctions() (map[string]*Function, error) {
	data, err := os.ReadFile(g.SourcePath)
	if err != nil {
		return nil, err
	}

	// modernc.org/cc is a C compiler frontend, but system SIMD intrinsic headers
	// (arm_neon.h, immintrin.h) are full of GNU/Clang extensions. We only need the
	// top-level function signatures from our file, so we parse a synthetic unit:
	//   - minimal typedefs/macros
	//   - extracted prototypes (no function bodies, no includes)
	parseUnit := buildParseUnit(string(data))

	cfg, err := cc.NewConfig(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return nil, err
	}

	ast, err := cc.Parse(cfg, []cc.Source{
		{Name: "<predefined>", Value: cfg.Predefined},
		{Name: "<builtin>", Value: cc.Builtin},
		{Name: g.SourcePath, Value: strings.NewReader(parseUnit)},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to parse C source %s: %w", g.SourcePath, err)
	}

	functions := make(map[string]*Function)
	for tu := ast.TranslationUnit; tu != nil; tu = tu.TranslationUnit {
		ex := tu.ExternalDeclaration
		if ex == nil {
			continue
		}
		if ex.Position().Filename != g.SourcePath {
			continue
		}
		switch ex.Case {
		case cc.ExternalDeclarationFuncDef:
			fd := ex.FunctionDefinition
			if fd == nil {
				continue
			}
			name, params, err := convertParams(fd.Declarator)
			if err != nil {
				return nil, err
			}
			ret := returnTypeString(fd.DeclarationSpecifiers)
			functions[name] = &Function{
				Name:       name,
				Parameters: params,
				ArgOffsets: make(map[string]int),
				ReturnType: ret,
			}
		case cc.ExternalDeclarationDecl:
			decl := ex.Declaration
			if decl == nil || decl.DeclarationSpecifiers == nil || decl.InitDeclaratorList == nil {
				continue
			}
			ret := returnTypeString(decl.DeclarationSpecifiers)
			for _, d := range flattenInitDeclarators(decl.InitDeclaratorList) {
				if d == nil || d.Declarator == nil {
					continue
				}
				name, params, err := convertParams(d.Declarator)
				if err != nil {
					// Not a function prototype.
					continue
				}
				functions[name] = &Function{
					Name:       name,
					Parameters: params,
					ArgOffsets: make(map[string]int),
					ReturnType: ret,
				}
			}
		}
	}

	return functions, nil
}

func flattenInitDeclarators(l *cc.InitDeclaratorList) []*cc.InitDeclarator {
	var out []*cc.InitDeclarator
	for cur := l; cur != nil; cur = cur.InitDeclaratorList {
		out = append(out, cur.InitDeclarator)
	}
	return out
}

func buildParseUnit(src string) string {
	prologue := strings.Join([]string{
		"// generated for signature parsing only",
		"#define __restrict__",
		"#define __attribute__(x)",
		"#define __builtin_popcount(x) 0",
		"typedef signed char int8_t;",
		"typedef unsigned char uint8_t;",
		"typedef unsigned short uint16_t;",
		"typedef long long int64_t;",
		"typedef unsigned long long uint64_t;",
		"typedef struct { long long _; } __m256i;",
		"typedef struct { long long _[8]; } __m512i;",
	}, "\n") + "\n\n"

	protos := extractTopLevelPrototypes(stripPreprocessorLines(stripComments(src)))
	return prologue + strings.Join(protos, "\n") + "\n"
}

func stripComments(s string) string {
	// Minimal comment stripper for C/C++ comments.
	// Good enough for our SIMD sources.
	var b strings.Builder
	b.Grow(len(s))
	inLine := false
	inBlock := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inLine {
			if c == '\n' {
				inLine = false
				b.WriteByte(c)
			}
			continue
		}
		if inBlock {
			if c == '*' && i+1 < len(s) && s[i+1] == '/' {
				inBlock = false
				i++
			}
			continue
		}
		if c == '/' && i+1 < len(s) {
			n := s[i+1]
			if n == '/' {
				inLine = true
				i++
				continue
			}
			if n == '*' {
				inBlock = true
				i++
				continue
			}
		}
		b.WriteByte(c)
	}
	return b.String()
}

func stripPreprocessorLines(s string) string {
	lines := strings.Split(s, "\n")
	out := lines[:0]
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

func extractTopLevelPrototypes(src string) []string {
	var protos []string
	braceDepth := 0
	lastBoundary := 0

	// Scan bytes; this intentionally ignores strings/char literals because our
	// SIMD sources don't embed braces in them.
	for i := 0; i < len(src); i++ {
		switch src[i] {
		case '{':
			if braceDepth == 0 {
				// Candidate top-level block. Only treat as function if the last
				// non-space char before '{' is ')'.
				j := i - 1
				for j >= 0 && (src[j] == ' ' || src[j] == '\t' || src[j] == '\n' || src[j] == '\r') {
					j--
				}
				if j >= 0 && src[j] == ')' {
					sig := strings.TrimSpace(src[lastBoundary:i])
					// Filter out non-function constructs.
					if sig != "" && !strings.HasPrefix(sig, "typedef") {
						protos = append(protos, sig+";")
					}
				}
			}
			braceDepth++
		case '}':
			braceDepth--
			if braceDepth == 0 {
				lastBoundary = i + 1
			}
		}
	}
	return protos
}

func returnTypeString(ds *cc.DeclarationSpecifiers) string {
	if ds == nil {
		return ""
	}
	// This mirrors GoAT's approach: for our purposes, distinguishing void/non-void is enough.
	if ds.Case == cc.DeclarationSpecifiersTypeSpec {
		return strings.TrimSpace(ds.TypeSpecifier.Token.SrcStr())
	}
	if ds.TypeSpecifier != nil {
		return strings.TrimSpace(ds.TypeSpecifier.Token.SrcStr())
	}
	return ""
}

func convertParams(decl *cc.Declarator) (string, []Parameter, error) {
	if decl == nil {
		return "", nil, fmt.Errorf("invalid function declarator")
	}
	dd := decl.DirectDeclarator
	if dd == nil || dd.Case != cc.DirectDeclaratorFuncParam {
		return "", nil, fmt.Errorf("invalid function parameter list")
	}

	fnName := dd.DirectDeclarator.Token.SrcStr()
	params, err := convertParamList(dd.ParameterTypeList.ParameterList)
	if err != nil {
		return "", nil, err
	}
	return fnName, params, nil
}

func convertParamList(pl *cc.ParameterList) ([]Parameter, error) {
	if pl == nil {
		return nil, nil
	}

	pd := pl.ParameterDeclaration
	if pd == nil {
		return nil, fmt.Errorf("invalid parameter declaration")
	}

	paramName := ""
	if pd.Declarator != nil && pd.Declarator.DirectDeclarator != nil {
		paramName = pd.Declarator.DirectDeclarator.Token.SrcStr()
	}
	if paramName == "" {
		// Unnamed parameter; keep a stable placeholder.
		paramName = "arg" + strconv.Itoa(pl.ParameterDeclaration.Position().Column)
	}

	paramType := ""
	if pd.DeclarationSpecifiers != nil {
		// Handle both TypeSpec and TypeQual cases.
		if pd.DeclarationSpecifiers.Case == cc.DeclarationSpecifiersTypeQual {
			paramType = pd.DeclarationSpecifiers.DeclarationSpecifiers.TypeSpecifier.Token.SrcStr()
		} else if pd.DeclarationSpecifiers.TypeSpecifier != nil {
			paramType = pd.DeclarationSpecifiers.TypeSpecifier.Token.SrcStr()
		}
	}
	isPointer := pd.Declarator != nil && pd.Declarator.Pointer != nil
	if isPointer {
		paramType = strings.TrimSpace(paramType) + "*"
	}

	params := []Parameter{{Name: paramName, Type: strings.TrimSpace(paramType)}}
	if pl.ParameterList != nil {
		next, err := convertParamList(pl.ParameterList)
		if err != nil {
			return nil, err
		}
		params = append(params, next...)
	}
	return params, nil
}
