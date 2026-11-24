package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"
	"net/http"
	"time"
)

// Defaults are the server-wide generation settings that a request may override.
type Defaults struct {
	Effort      float64
	Budget      int
	Temperature float64
	TopP        float64
	TopK        int
	MaxTokens   int // 0: up to the end of the context
}

// Server serves an OpenAI-style API for one model.
type Server struct {
	Model    string // the model id clients send and see
	Engine   *Engine
	Defaults Defaults
	Log      *log.Logger
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-s.Engine.Dead():
			http.Error(w, "engine stopped", http.StatusServiceUnavailable)
		default:
			_, _ = w.Write([]byte("ok\n"))
		}
	})
	return mux
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []any{
		map[string]any{"id": s.Model, "object": "model", "owned_by": "local", "context_length": s.Engine.Ctx},
	}})
}

// chatRequest holds the fields this server reads; others are ignored.
type chatRequest struct {
	Model         string          `json:"model"`
	Messages      []Message       `json:"messages"`
	Tools         []Tool          `json:"tools"`
	ToolChoice    json.RawMessage `json:"tool_choice"`
	Stream        bool            `json:"stream"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"`
	Seed                *uint64         `json:"seed"`
	Stop                json.RawMessage `json:"stop"`
	ReasoningEffort     any             `json:"reasoning_effort"`
	ChatTemplateKwargs  struct {
		ReasoningEffort any `json:"reasoning_effort"`
	} `json:"chat_template_kwargs"`
	// thinking budget: thinking_budget (omlx) or thinking_budget_tokens (LM Studio); 0 or less turns it off
	ThinkingBudget       *int `json:"thinking_budget"`
	ThinkingBudgetTokens *int `json:"thinking_budget_tokens"`
}

type apiError struct {
	status int
	msg    string
	code   string
}

func (e *apiError) Error() string { return e.msg }

func badRequest(format string, a ...any) *apiError {
	return &apiError{status: http.StatusBadRequest, msg: fmt.Sprintf(format, a...)}
}

// prepare validates a request and turns it into a prompt and generation settings.
func (s *Server) prepare(req *chatRequest) (string, GenParams, []string, *apiError) {
	if len(req.Messages) == 0 {
		return "", GenParams{}, nil, badRequest("messages must not be empty")
	}
	if req.Model != "" && req.Model != s.Model {
		return "", GenParams{}, nil, &apiError{status: http.StatusNotFound, msg: fmt.Sprintf("model %q not found; this server has %q", req.Model, s.Model), code: "model_not_found"}
	}
	if tc := string(req.ToolChoice); tc != "" && tc != "null" && tc != `"auto"` && tc != `"none"` {
		return "", GenParams{}, nil, badRequest("tool_choice %s is not supported (only auto and none)", tc)
	}
	tools := req.Tools
	if string(req.ToolChoice) == `"none"` {
		tools = nil
	}
	effort := s.Defaults.Effort
	for _, v := range []any{req.ChatTemplateKwargs.ReasoningEffort, req.ReasoningEffort} {
		if v != nil {
			f, err := ParseEffort(v)
			if err != nil {
				return "", GenParams{}, nil, badRequest("%v", err)
			}
			effort = f
		}
	}
	prompt, err := Render(req.Messages, tools, effort)
	if err != nil {
		return "", GenParams{}, nil, badRequest("%v", err)
	}
	p := GenParams{MaxTokens: s.Defaults.MaxTokens, Temp: s.Defaults.Temperature, TopP: s.Defaults.TopP, TopK: s.Defaults.TopK,
		Seed: rand.Uint64(), Budget: s.Defaults.Budget}
	for _, v := range []*int{req.MaxTokens, req.MaxCompletionTokens} {
		if v != nil {
			if *v < 1 {
				return "", GenParams{}, nil, badRequest("max_tokens must be at least 1")
			}
			p.MaxTokens = *v
		}
	}
	if p.MaxTokens <= 0 {
		p.MaxTokens = s.Engine.Ctx
	}
	if req.Temperature != nil {
		p.Temp = *req.Temperature
	}
	if req.TopP != nil {
		p.TopP = *req.TopP
	}
	if req.TopK != nil {
		p.TopK = max(*req.TopK, 0)
	}
	if req.Seed != nil {
		p.Seed = *req.Seed
	}
	for _, v := range []*int{req.ThinkingBudget, req.ThinkingBudgetTokens} {
		if v != nil {
			p.Budget = max(*v, 0)
		}
	}
	stops, err := parseStop(req.Stop)
	if err != nil {
		return "", GenParams{}, nil, badRequest("%v", err)
	}
	return prompt, p, stops, nil
}

func parseStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, errors.New("stop must be a string or a list of strings")
	}
	return many, nil
}

type usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func makeUsage(r GenResult) usage {
	u := usage{PromptTokens: r.PromptTokens, CompletionTokens: r.GenTokens, TotalTokens: r.PromptTokens + r.GenTokens}
	u.PromptTokensDetails.CachedTokens = r.CachedTokens
	u.CompletionTokensDetails.ReasoningTokens = r.ThinkTokens
	return u
}

func finishReason(r GenResult, p *Parser) string {
	switch {
	case r.Reason == "length":
		return "length"
	case len(p.ToolCalls) > 0 && !p.Stopped:
		return "tool_calls"
	}
	return "stop"
}

func contextError(r GenResult, ctx int) *apiError {
	return &apiError{status: http.StatusBadRequest, code: "context_length_exceeded",
		msg: fmt.Sprintf("This model's maximum context length is %d tokens. However, your messages resulted in %d tokens.", ctx, r.PromptTokens)}
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<20)).Decode(&req); err != nil {
		writeError(w, badRequest("invalid JSON: %v", err))
		return
	}
	prompt, params, stops, aerr := s.prepare(&req)
	if aerr != nil {
		writeError(w, aerr)
		return
	}
	id := "chatcmpl-" + newCallID()[5:]
	created := time.Now().Unix()
	parser := NewParser(stops)
	start := time.Now()
	if req.Stream {
		s.stream(w, r.Context(), &req, id, created, prompt, params, parser, start)
		return
	}
	res, err := s.Engine.Generate(r.Context(), prompt, params, nil, func(special bool, b []byte) error {
		parser.Push(special, b)
		if parser.Stopped {
			return ErrStop
		}
		return nil
	})
	parser.Push(true, []byte(tokEnd)) // flush held-back text
	s.logRequest(res, err, start, parser)
	if err != nil {
		writeError(w, &apiError{status: http.StatusInternalServerError, msg: err.Error()})
		return
	}
	if res.Reason == "context" {
		writeError(w, contextError(res, s.Engine.Ctx))
		return
	}
	msg := map[string]any{"role": "assistant", "content": parser.Content.String()}
	if parser.Reasoning.Len() > 0 {
		msg["reasoning_content"] = parser.Reasoning.String()
	}
	if len(parser.ToolCalls) > 0 {
		msg["tool_calls"] = parser.ToolCalls
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": s.Model,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finishReason(res, parser)}},
		"usage":   makeUsage(res),
	})
}

func (s *Server) stream(w http.ResponseWriter, ctx context.Context, req *chatRequest, id string, created int64,
	prompt string, params GenParams, parser *Parser, start time.Time) {
	fl, _ := w.(http.Flusher)
	started := false
	send := func(v any) error {
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(http.StatusOK)
			started = true
		}
		b, _ := json.Marshal(v)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return err
		}
		if fl != nil {
			fl.Flush()
		}
		return nil
	}
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.Model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	nCalls := 0
	emit := func(ds []Delta) error {
		for _, d := range ds {
			var delta map[string]any
			switch {
			case d.Reasoning != "":
				delta = map[string]any{"reasoning_content": d.Reasoning}
			case d.Content != "":
				delta = map[string]any{"content": d.Content}
			case d.ToolCall != nil:
				delta = map[string]any{"tool_calls": []any{map[string]any{"index": nCalls, "id": d.ToolCall.ID, "type": "function",
					"function": map[string]any{"name": d.ToolCall.Function.Name, "arguments": d.ToolCall.Function.Arguments}}}}
				nCalls++
			default:
				continue
			}
			if err := send(chunk(delta, nil)); err != nil {
				return err
			}
		}
		return nil
	}
	var res GenResult
	var err error
	res, err = s.Engine.Generate(ctx, prompt, params, func(pr GenResult) {
		if pr.Reason == "" && pr.PromptTokens+2 < s.Engine.Ctx {
			_ = send(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
		}
	}, func(special bool, b []byte) error {
		if e := emit(parser.Push(special, b)); e != nil {
			return e // the client is gone
		}
		if parser.Stopped {
			return ErrStop
		}
		return nil
	})
	_ = emit(parser.Push(true, []byte(tokEnd)))
	s.logRequest(res, err, start, parser)
	if err == nil && res.Reason == "context" {
		if started {
			_ = send(map[string]any{"error": contextError(res, s.Engine.Ctx).body()})
		} else {
			writeError(w, contextError(res, s.Engine.Ctx))
		}
		return
	}
	if err != nil {
		if !started {
			writeError(w, &apiError{status: http.StatusInternalServerError, msg: err.Error()})
		} else if ctx.Err() == nil {
			_ = send(map[string]any{"error": map[string]any{"message": err.Error(), "type": "server_error"}})
		}
		return
	}
	_ = send(chunk(map[string]any{}, finishReason(res, parser)))
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		_ = send(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.Model,
			"choices": []any{}, "usage": makeUsage(res)})
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if fl != nil {
		fl.Flush()
	}
}

func (s *Server) logRequest(res GenResult, err error, start time.Time, p *Parser) {
	tps := 0.0
	if res.Gen > 0 {
		tps = float64(res.GenTokens) / res.Gen.Seconds()
	}
	status := res.Reason
	if err != nil {
		status = "error: " + err.Error()
	}
	s.Log.Printf("prompt=%d cached=%d prefill=%.2fs gen=%d think=%d decode=%.1ftok/s tools=%d total=%.1fs end=%s",
		res.PromptTokens, res.CachedTokens, res.Prefill.Seconds(), res.GenTokens, res.ThinkTokens, tps, len(p.ToolCalls),
		time.Since(start).Seconds(), status)
	if p.Malformed > 0 {
		s.Log.Printf("warning: %d tool-call message(s) did not parse; passed on as text", p.Malformed)
	}
}

func (e *apiError) body() map[string]any {
	typ := "invalid_request_error"
	if e.status >= 500 {
		typ = "server_error"
	}
	b := map[string]any{"message": e.msg, "type": typ}
	if e.code != "" {
		b["code"] = e.code
	}
	return b
}

func writeError(w http.ResponseWriter, e *apiError) {
	writeJSON(w, e.status, map[string]any{"error": e.body()})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}
