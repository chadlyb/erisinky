package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// The model writes its reply as a sequence of messages:
//
//	<|message_model|><|content_thinking|>…<|end_message|>          reasoning
//	<|message_model|><|content_text|>…<|end_message|>              the answer
//	<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{…}}<|end_message|>   a tool call
//
// and ends with <|content_model_end_sampling|>, which the engine treats as its stop token. Parser turns that token
// stream into reasoning, content and tool calls.

// Delta is what one token adds to the reply. At most one field is set.
type Delta struct {
	Reasoning string
	Content   string
	ToolCall  *ToolCall // complete, once its message ends
}

type section int

const (
	secNone section = iota
	secHeader
	secThinking
	secText
	secToolJSON
	secOther // a content type this server doesn't handle; its text is dropped
)

// Parser consumes engine tokens and emits deltas. Text is emitted only as whole UTF-8 characters.
type Parser struct {
	sec     section
	pending []byte // bytes of a split UTF-8 character
	header  strings.Builder
	toolBuf bytes.Buffer
	calls   int
	// Stop strings cut the answer text. held keeps back text that might be the start of one.
	stops   []string
	held    string
	Stopped bool
	// Content and Reasoning collect everything emitted, for non-streaming replies.
	Content   strings.Builder
	Reasoning strings.Builder
	ToolCalls []ToolCall
	// Malformed counts tool-call messages whose JSON didn't parse; their text goes to Content instead.
	Malformed int
}

// NewParser returns a parser that stops the answer at the first of stops.
func NewParser(stops []string) *Parser {
	var s []string
	for _, x := range stops {
		if x != "" {
			s = append(s, x)
		}
	}
	return &Parser{stops: s}
}

// Push feeds one token. special marks control tokens.
func (p *Parser) Push(special bool, b []byte) []Delta {
	if special {
		return p.control(string(b))
	}
	switch p.sec {
	case secHeader:
		p.header.Write(b)
		return nil
	case secToolJSON:
		p.toolBuf.Write(b)
		return nil
	case secThinking, secText:
		s := p.text(b)
		if s == "" {
			return nil
		}
		if p.sec == secThinking {
			p.Reasoning.WriteString(s)
			return []Delta{{Reasoning: s}}
		}
		return p.content(s)
	}
	return nil // text outside a message, or in an unhandled section
}

func (p *Parser) control(tok string) []Delta {
	var out []Delta
	switch tok {
	case tokModel:
		out = p.flushText()
		p.sec = secHeader
		p.header.Reset()
	case tokThinking:
		p.sec = secThinking
	case tokText:
		p.sec = secText
	case tokInvokeJSON:
		p.sec = secToolJSON
		p.toolBuf.Reset()
	case tokEnd:
		out = p.flushText()
		if p.sec == secToolJSON {
			out = append(out, p.finishTool()...)
		}
		p.sec = secNone
	default:
		if strings.HasPrefix(tok, "<|content_") {
			p.sec = secOther
		}
	}
	return out
}

// text turns token bytes into whole characters, keeping a split character for the next token.
func (p *Parser) text(b []byte) string {
	p.pending = append(p.pending, b...)
	n := len(p.pending)
	// keep back an incomplete character at the end (at most 3 bytes)
	for i := 1; i <= 3 && i <= n; i++ {
		c := p.pending[n-i]
		if c&0xC0 != 0x80 { // a start byte: is its character complete?
			if !utf8.FullRune(p.pending[n-i:]) {
				n -= i
			}
			break
		}
	}
	s := strings.ToValidUTF8(string(p.pending[:n]), "�")
	p.pending = append(p.pending[:0], p.pending[n:]...)
	return s
}

// flushText ends a text section: emits any held-back text and drops a broken trailing character.
func (p *Parser) flushText() []Delta {
	p.pending = p.pending[:0]
	if p.held == "" || p.Stopped {
		return nil
	}
	s := p.held
	p.held = ""
	p.Content.WriteString(s)
	return []Delta{{Content: s}}
}

// content applies the stop strings to answer text.
func (p *Parser) content(s string) []Delta {
	if p.Stopped {
		return nil
	}
	if len(p.stops) == 0 {
		p.Content.WriteString(s)
		return []Delta{{Content: s}}
	}
	buf := p.held + s
	cut := -1
	for _, st := range p.stops {
		if i := strings.Index(buf, st); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut >= 0 {
		p.Stopped = true
		p.held = ""
		if cut == 0 {
			return nil
		}
		p.Content.WriteString(buf[:cut])
		return []Delta{{Content: buf[:cut]}}
	}
	// hold back the longest tail that is a prefix of some stop string
	keep := 0
	for _, st := range p.stops {
		for k := min(len(st)-1, len(buf)); k > keep; k-- {
			if strings.HasSuffix(buf, st[:k]) {
				keep = k
				break
			}
		}
	}
	emit := buf[:len(buf)-keep]
	p.held = buf[len(buf)-keep:]
	if emit == "" {
		return nil
	}
	p.Content.WriteString(emit)
	return []Delta{{Content: emit}}
}

func (p *Parser) finishTool() []Delta {
	name := strings.TrimSpace(p.header.String())
	var body struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	}
	raw := p.toolBuf.Bytes()
	if err := json.Unmarshal(raw, &body); err != nil || (body.Name == "" && name == "") {
		// not a usable call: pass the text on, so the client sees what the model wrote
		p.Malformed++
		return p.content(string(raw))
	}
	if body.Name == "" {
		body.Name = name
	}
	args := "{}"
	if a := bytes.TrimSpace(body.Args); len(a) > 0 && string(a) != "null" {
		args = string(a)
	}
	tc := ToolCall{ID: newCallID(), Type: "function", Function: FunctionCall{Name: body.Name, Arguments: args}}
	p.calls++
	p.ToolCalls = append(p.ToolCalls, tc)
	return []Delta{{ToolCall: &tc}}
}

func newCallID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("crypto/rand: %v", err))
	}
	return "call_" + hex.EncodeToString(b[:])
}
