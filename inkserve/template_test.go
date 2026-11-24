package main

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// TestRenderMatchesTemplate compares Render with the model's own chat template, rendered by transformers into
// testdata/golden.json (see testdata/golden.py).
func TestRenderMatchesTemplate(t *testing.T) {
	var cases []struct {
		Name     string    `json:"name"`
		Effort   any       `json:"effort"`
		Messages []Message `json:"messages"`
		Tools    []Tool    `json:"tools"`
	}
	var golden []struct {
		Name   string `json:"name"`
		Prompt string `json:"prompt"`
	}
	readJSON(t, "testdata/cases.json", &cases)
	readJSON(t, "testdata/golden.json", &golden)
	if len(cases) != len(golden) {
		t.Fatalf("%d cases but %d golden prompts: regenerate golden.json", len(cases), len(golden))
	}
	for i, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			eff, err := ParseEffort(c.Effort)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Render(c.Messages, c.Tools, eff)
			if err != nil {
				t.Fatal(err)
			}
			if want := golden[i].Prompt; got != want {
				n := commonPrefix(got, want)
				t.Errorf("differs at byte %d\n got: …%s\nwant: …%s", n, clip(got[n:]), clip(want[n:]))
			}
		})
	}
}

func TestRenderRejects(t *testing.T) {
	for _, tc := range []struct {
		name string
		msgs []Message
	}{
		{"unknown role", []Message{{Role: "developer", Content: json.RawMessage(`"x"`)}}},
		{"image", []Message{{Role: "user", Content: json.RawMessage(`[{"type":"image_url","image_url":{"url":"x"}}]`)}}},
		{"string arguments that are not an object", []Message{{Role: "assistant", ToolCalls: []ToolCall{{Function: FunctionCall{Name: "f", Arguments: `[1]`}}}}}},
	} {
		if _, err := Render(tc.msgs, nil, 0.9); err == nil {
			t.Errorf("%s: no error", tc.name)
		}
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func commonPrefix(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

func clip(s string) string {
	if len(s) > 120 {
		return s[:120]
	}
	return strings.ReplaceAll(s, "\n", `\n`)
}

func TestPyNumber(t *testing.T) {
	for in, want := range map[string]string{
		"2": "2", "-0": "0", "12345678901234567890": "12345678901234567890",
		"2.50": "2.5", "1.0": "1.0", "1e3": "1000.0", "1e16": "1e+16", "1.5e-7": "1.5e-07", "0.0001": "0.0001",
		"0.00001": "1e-05", "1234567.0": "1234567.0", "-3.25": "-3.25",
	} {
		if got := pyNumber(in); got != want {
			t.Errorf("pyNumber(%s) = %s, want %s", in, got, want)
		}
	}
}
