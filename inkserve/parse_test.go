package main

import (
	"strings"
	"testing"
)

// tok is one engine token for the tests: a control token, or text bytes.
type tok struct {
	special bool
	b       string
}

func ctl(s string) tok { return tok{true, s} }
func txt(s string) tok { return tok{false, s} }

func run(p *Parser, toks []tok) []Delta {
	var out []Delta
	for _, t := range toks {
		out = append(out, p.Push(t.special, []byte(t.b))...)
	}
	return out
}

func TestParserReasoningAndText(t *testing.T) {
	p := NewParser(nil)
	run(p, []tok{ctl(tokModel), ctl(tokThinking), txt("Think"), txt("ing."), ctl(tokEnd),
		ctl(tokModel), ctl(tokText), txt("The "), txt("answer."), ctl(tokEnd)})
	if got := p.Reasoning.String(); got != "Thinking." {
		t.Errorf("reasoning %q", got)
	}
	if got := p.Content.String(); got != "The answer." {
		t.Errorf("content %q", got)
	}
}

func TestParserToolCall(t *testing.T) {
	p := NewParser(nil)
	ds := run(p, []tok{ctl(tokModel), txt("run_"), txt("shell"), ctl(tokInvokeJSON),
		txt(`{"name":"run_shell",`), txt(`"args":{"cmd":"ls"}}`), ctl(tokEnd)})
	if len(p.ToolCalls) != 1 {
		t.Fatalf("tool calls: %+v", p.ToolCalls)
	}
	c := p.ToolCalls[0]
	if c.Function.Name != "run_shell" || c.Function.Arguments != `{"cmd":"ls"}` || !strings.HasPrefix(c.ID, "call_") {
		t.Errorf("call %+v", c)
	}
	if len(ds) != 1 || ds[0].ToolCall == nil {
		t.Errorf("deltas %+v", ds)
	}
}

func TestParserMalformedToolCallBecomesText(t *testing.T) {
	p := NewParser(nil)
	run(p, []tok{ctl(tokModel), txt("f"), ctl(tokInvokeJSON), txt(`{"name": oops`), ctl(tokEnd)})
	if len(p.ToolCalls) != 0 || p.Malformed != 1 || p.Content.String() != `{"name": oops` {
		t.Errorf("calls %v malformed %d content %q", p.ToolCalls, p.Malformed, p.Content.String())
	}
}

func TestParserSplitUTF8(t *testing.T) {
	p := NewParser(nil)
	e := "é" // 0xC3 0xA9
	ds := run(p, []tok{ctl(tokModel), ctl(tokText), txt("caf"), txt(e[:1]), txt(e[1:] + "!"), ctl(tokEnd)})
	for _, d := range ds {
		if strings.ContainsRune(d.Content, '�') {
			t.Errorf("broken character in delta %q", d.Content)
		}
	}
	if got := p.Content.String(); got != "café!" {
		t.Errorf("content %q", got)
	}
}

func TestParserStopString(t *testing.T) {
	p := NewParser([]string{"STOP"})
	var streamed strings.Builder
	for _, d := range run(p, []tok{ctl(tokModel), ctl(tokText), txt("abc S"), txt("TO"), txt("P tail")}) {
		streamed.WriteString(d.Content)
	}
	if !p.Stopped || p.Content.String() != "abc " || streamed.String() != "abc " {
		t.Errorf("stopped %v content %q streamed %q", p.Stopped, p.Content.String(), streamed.String())
	}
	// a partial match that doesn't complete is released at the end of the message
	q := NewParser([]string{"STOP"})
	run(q, []tok{ctl(tokModel), ctl(tokText), txt("ST"), ctl(tokEnd)})
	if q.Stopped || q.Content.String() != "ST" {
		t.Errorf("stopped %v content %q", q.Stopped, q.Content.String())
	}
}

func TestParserIgnoresUnknownContent(t *testing.T) {
	p := NewParser(nil)
	run(p, []tok{ctl(tokModel), ctl("<|content_image|>"), txt("x"), ctl(tokEnd), ctl(tokModel), ctl(tokText), txt("ok"), ctl(tokEnd)})
	if p.Content.String() != "ok" {
		t.Errorf("content %q", p.Content.String())
	}
}
