package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// This file renders an OpenAI-style conversation into Inkling 2 Mini's prompt format. It follows the model's own
// chat_template.jinja (shipped with the model) line for line, so that prompts match what the model was trained on
// and a re-rendered history matches the cached prefix.

// Message is one chat message as OpenAI-style clients send it.
type Message struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content,omitempty"` // a string, an array of parts, or null
	Name             string          `json:"name,omitempty"`
	ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
}

// ToolCall is one function call in an assistant message.
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall names a function and carries its arguments as a JSON object encoded in a string.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool declares a function the model may call.
type Tool struct {
	Type     string          `json:"type,omitempty"`
	Function json.RawMessage `json:"function"`
}

const (
	tokUser        = "<|message_user|>"
	tokModel       = "<|message_model|>"
	tokSystem      = "<|message_system|>"
	tokTool        = "<|message_tool|>"
	tokText        = "<|content_text|>"
	tokThinking    = "<|content_thinking|>"
	tokInvokeJSON  = "<|content_invoke_tool_json|>"
	tokXML         = "<|content_xml|>"
	tokEnd         = "<|end_message|>"
	tokEndSampling = "<|content_model_end_sampling|>"
)

var roleToken = map[string]string{"user": tokUser, "assistant": tokModel, "system": tokSystem, "tool": tokTool}

// effortLevels maps reasoning_effort names to the numbers the template writes.
var effortLevels = map[string]float64{"none": 0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}

// ParseEffort reads reasoning_effort: a name from effortLevels or a number in [0, 0.99].
func ParseEffort(v any) (float64, error) {
	switch x := v.(type) {
	case string:
		key := strings.TrimSpace(x)
		if f, ok := effortLevels[key]; ok {
			return f, nil
		}
		if f, err := strconv.ParseFloat(key, 64); err == nil {
			return checkEffort(f)
		}
		return 0, fmt.Errorf("unknown reasoning_effort %q", x)
	case float64:
		return checkEffort(x)
	}
	return 0, fmt.Errorf("reasoning_effort must be a string or a number")
}

func checkEffort(f float64) (float64, error) {
	if f < 0 || f > 0.99 {
		return 0, errors.New("reasoning_effort must be in [0.0, 0.99]")
	}
	return f, nil
}

// formatEffort prints the number as Jinja does: "0" for zero, otherwise Python's shortest float form.
func formatEffort(f float64) string {
	if f == 0 {
		return "0"
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// Render builds the prompt for messages, tools and a thinking effort. It returns a client error for input the
// template rejects.
func Render(msgs []Message, tools []Tool, effort float64) (string, error) {
	var b strings.Builder
	if len(tools) > 0 {
		specs, err := toolSpecs(tools)
		if err != nil {
			return "", err
		}
		b.WriteString(tokSystem + "tool_declare" + tokXML + specs + tokEnd)
	}
	effortDone := false
	emitEffort := func() {
		b.WriteString(tokSystem + tokText + "Thinking effort level: " + formatEffort(effort) + tokEnd)
		effortDone = true
	}
	for i, m := range msgs {
		rt, ok := roleToken[m.Role]
		if !ok {
			return "", fmt.Errorf("unknown message role %q", m.Role)
		}
		if !effortDone && m.Role != "system" {
			emitEffort()
		}
		header := rt
		switch {
		case m.Role == "tool" && m.Name != "":
			header += m.Name
		case m.Role == "tool" && m.ToolCallID != "":
			header += toolNameFor(msgs, m.ToolCallID)
		case m.Role == "assistant" && m.Name != "":
			header += m.Name
		}
		if m.Role == "assistant" && m.ReasoningContent != "" {
			b.WriteString(header + tokThinking + m.ReasoningContent + tokEnd)
		}
		if err := renderContent(&b, m, header); err != nil {
			return "", fmt.Errorf("message %d: %w", i, err)
		}
		if m.Role == "assistant" {
			for _, tc := range m.ToolCalls {
				if m.Name != "" && m.Name != tc.Function.Name {
					return "", fmt.Errorf("message %d: assistant name %s does not match its tool call %s", i, m.Name, tc.Function.Name)
				}
				args, err := canonicalArgs(tc.Function.Arguments)
				if err != nil {
					return "", fmt.Errorf("message %d: tool call %s: %w", i, tc.Function.Name, err)
				}
				b.WriteString(tokModel + tc.Function.Name + tokInvokeJSON + `{"name":` + jsonString(tc.Function.Name) + `,"args":` + args + "}" + tokEnd)
			}
			b.WriteString(tokEndSampling)
		}
	}
	if !effortDone {
		emitEffort()
	}
	return b.String(), nil
}

// toolNameFor finds the function name of the tool call with this id. Like the template, the last match wins.
func toolNameFor(msgs []Message, id string) string {
	name := ""
	for _, p := range msgs {
		if p.Role != "assistant" {
			continue
		}
		for _, tc := range p.ToolCalls {
			if tc.ID == id && tc.Function.Name != "" {
				name = tc.Function.Name
			}
		}
	}
	return name
}

type contentPart struct {
	Type string  `json:"type"`
	Text *string `json:"text"`
}

func renderContent(b *strings.Builder, m Message, header string) error {
	c := bytes.TrimSpace(m.Content)
	if len(c) == 0 || string(c) == "null" {
		return nil // absent or null content: the template writes no text block
	}
	if c[0] == '"' {
		var s string
		if err := json.Unmarshal(c, &s); err != nil {
			return fmt.Errorf("content: %w", err)
		}
		if s != "" || !(m.Role == "assistant" && (len(m.ToolCalls) > 0 || m.ReasoningContent != "")) {
			b.WriteString(header + tokText + s + tokEnd)
		}
		return nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(c, &parts); err != nil {
		return fmt.Errorf("content must be a string or an array of parts")
	}
	for _, raw := range parts {
		if len(raw) > 0 && raw[0] == '"' {
			var s string
			_ = json.Unmarshal(raw, &s)
			b.WriteString(header + tokText + s + tokEnd)
			continue
		}
		var p contentPart
		if err := json.Unmarshal(raw, &p); err != nil {
			return fmt.Errorf("content part: %w", err)
		}
		switch p.Type {
		case "", "text", "input_text":
			t := ""
			if p.Text != nil {
				t = *p.Text
			}
			b.WriteString(header + tokText + t + tokEnd)
		case "image", "input_image", "image_url":
			return errors.New("image input is not supported by this server")
		case "audio", "input_audio", "audio_url":
			return errors.New("Inkling 2 does not accept audio input")
		default:
			return fmt.Errorf("unsupported content part type %q", p.Type)
		}
	}
	return nil
}

// toolSpecs writes the tool declarations as the template does: a list of {description, name, parameters, type}
// with every key sorted and no spaces.
func toolSpecs(tools []Tool) (string, error) {
	specs := make([]any, 0, len(tools))
	for i, t := range tools {
		fn := t.Function
		if len(fn) == 0 {
			return "", fmt.Errorf("tool %d has no function", i)
		}
		var f struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		}
		if err := json.Unmarshal(fn, &f); err != nil {
			return "", fmt.Errorf("tool %d: %w", i, err)
		}
		var params any = map[string]any{}
		if p := bytes.TrimSpace(f.Parameters); len(p) > 0 && string(p) != "null" {
			d := json.NewDecoder(bytes.NewReader(p))
			d.UseNumber()
			if err := d.Decode(&params); err != nil {
				return "", fmt.Errorf("tool %d parameters: %w", i, err)
			}
			params = normalizeNumbers(params)
		}
		typ := t.Type
		if typ == "" {
			typ = "function"
		}
		specs = append(specs, map[string]any{"description": f.Description, "name": f.Name, "parameters": params, "type": typ})
	}
	return marshalSorted(specs), nil
}

// pyNumber prints a JSON number literal as Python's json.dumps(json.loads(s)) does: integers as written (Python
// keeps them exact), everything else as a float in Python's repr form (2.50 -> 2.5, 1e3 -> 1000.0, 1e16 -> 1e+16).
func pyNumber(s string) string {
	if !strings.ContainsAny(s, ".eE") {
		if strings.Trim(s, "-0") == "" {
			return "0"
		}
		return s
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return s
	}
	e := strconv.FormatFloat(f, 'e', -1, 64) // shortest digits, as Python's repr
	exp, _ := strconv.Atoi(e[strings.IndexByte(e, 'e')+1:])
	if exp < -4 || exp >= 16 {
		return e // Go and Python agree here: 1e+16, 1.5e-07
	}
	fixed := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(fixed, ".") {
		fixed += ".0"
	}
	return fixed
}

// normalizeNumbers rewrites every json.Number in a decoded value with pyNumber.
func normalizeNumbers(v any) any {
	switch x := v.(type) {
	case json.Number:
		return json.Number(pyNumber(x.String()))
	case map[string]any:
		for k, e := range x {
			x[k] = normalizeNumbers(e)
		}
	case []any:
		for i, e := range x {
			x[i] = normalizeNumbers(e)
		}
	}
	return v
}

// marshalSorted encodes v compactly with sorted keys and without HTML escaping (encoding/json sorts map keys).
func marshalSorted(v any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
	return strings.TrimSuffix(buf.String(), "\n")
}

func jsonString(s string) string { return marshalSorted(s) }

// canonicalArgs turns a tool call's arguments (a JSON object in a string) into the template's form: top-level keys
// sorted, each value compact with its own key order kept.
func canonicalArgs(s string) (string, error) {
	if strings.TrimSpace(s) == "" {
		return "{}", nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(s), &obj); err != nil {
		return "", fmt.Errorf("arguments must be a JSON object: %w", err)
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys) // byte order, as Python's dictsort(case_sensitive=True) for these strings
	var b strings.Builder
	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		v, err := compactOrdered(obj[k])
		if err != nil {
			return "", err
		}
		b.WriteString(jsonString(k) + ":" + v)
	}
	b.WriteByte('}')
	return b.String(), nil
}

// compactOrdered re-encodes a JSON value without spaces, keeping object key order and number literals, and writing
// strings as Python's json.dumps(ensure_ascii=False) does (non-ASCII as is).
func compactOrdered(raw json.RawMessage) (string, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var b strings.Builder
	type frame struct {
		obj   bool
		first bool
		key   bool // in an object: the next string is a key
	}
	var stack []frame
	for {
		t, err := d.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		// separators: before a key or an array element after the first, ','; between a key and its value, ':'
		if n := len(stack); n > 0 && t != json.Delim('}') && t != json.Delim(']') {
			f := &stack[n-1]
			switch {
			case f.obj && f.key && !f.first, !f.obj && !f.first:
				b.WriteByte(',')
			case f.obj && !f.key:
				b.WriteByte(':')
			}
			if !f.obj || !f.key {
				f.first = false
			}
			if f.obj {
				f.key = !f.key
			}
		}
		switch v := t.(type) {
		case json.Delim:
			b.WriteRune(rune(v))
			switch v {
			case '{':
				stack = append(stack, frame{obj: true, first: true, key: true})
			case '[':
				stack = append(stack, frame{first: true})
			default:
				stack = stack[:len(stack)-1]
			}
		case string:
			b.WriteString(jsonString(v))
		case json.Number:
			b.WriteString(pyNumber(v.String()))
		case bool:
			b.WriteString(strconv.FormatBool(v))
		case nil:
			b.WriteString("null")
		}
	}
	if !utf8.ValidString(b.String()) {
		return "", errors.New("arguments are not valid UTF-8")
	}
	return b.String(), nil
}
