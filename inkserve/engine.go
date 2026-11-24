package main

import (
	"bufio"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Engine drives one `inkling serve` process over its stdin and stdout (protocol: src/serve.rs at the repository root). It runs one
// request at a time; callers queue on its lock.
type Engine struct {
	Ctx   int // context length in tokens
	Vocab int

	lock chan struct{} // a one-slot semaphore, so waiting callers can give up when their context ends
	in   io.WriteCloser
	out  *bufio.Reader
	cmd  *exec.Cmd
	dead chan struct{} // closed when the process exits
	err  error         // why it exited
	mu   sync.Mutex
}

// GenParams are the per-request generation settings.
type GenParams struct {
	MaxTokens int
	Temp      float64
	TopP      float64
	TopK      int
	Seed      uint64
	Budget    int // thinking tokens; 0 = no limit
}

// GenResult summarizes one generation.
type GenResult struct {
	PromptTokens int
	CachedTokens int
	Prefill      time.Duration
	GenTokens    int
	ThinkTokens  int
	Gen          time.Duration
	Reason       string // stop | length | cancel | context
}

// ErrStop, returned by a token callback, ends the generation early (for example at a stop string).
var ErrStop = errors.New("stop requested")

// StartEngine runs the engine binary and waits until the model is loaded. A non-empty drafter (the DSpark GGUF) turns
// on speculative decoding (DSPARK.md).
func StartEngine(bin, model, drafter string, ctx int, stderr io.Writer) (*Engine, error) {
	args := []string{"serve", "-m", model, "-c", strconv.Itoa(ctx)}
	if drafter != "" {
		args = append(args, "--ref", drafter)
	}
	cmd := exec.Command(bin, args...)
	cmd.Stderr = stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start engine: %w", err)
	}
	e := &Engine{lock: make(chan struct{}, 1), in: in, out: bufio.NewReaderSize(outPipe, 1<<16), cmd: cmd, dead: make(chan struct{})}
	go func() {
		err := cmd.Wait()
		e.mu.Lock()
		e.err = fmt.Errorf("engine exited: %v", err)
		e.mu.Unlock()
		close(e.dead)
	}()
	line, err := e.out.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("engine failed to load the model (see its log): %w", err)
	}
	f := strings.Fields(line)
	if len(f) < 3 || f[0] != "R" {
		return nil, fmt.Errorf("engine: unexpected first line %q", line)
	}
	kv := fields(f[1:])
	e.Ctx, _ = strconv.Atoi(kv["ctx"])
	e.Vocab, _ = strconv.Atoi(kv["vocab"])
	return e, nil
}

// Dead is closed when the engine process exits.
func (e *Engine) Dead() <-chan struct{} { return e.dead }

// Close stops the engine.
func (e *Engine) Close() error {
	_ = e.in.Close() // the engine exits at end of input
	select {
	case <-e.dead:
	case <-time.After(10 * time.Second):
		_ = e.cmd.Process.Kill()
		<-e.dead
	}
	return nil
}

// Generate runs one prompt. onToken gets each generated token; returning ErrStop ends the reply, any other error
// aborts it. If ctx ends (the client went away) the generation stops.
func (e *Engine) Generate(ctx context.Context, prompt string, p GenParams, onPrefill func(GenResult), onToken func(special bool, b []byte) error) (GenResult, error) {
	select {
	case e.lock <- struct{}{}:
	case <-ctx.Done():
		return GenResult{}, ctx.Err()
	case <-e.dead:
		return GenResult{}, e.deadErr()
	}
	defer func() { <-e.lock }()

	req := fmt.Sprintf("G max=%d temp=%s top_p=%s top_k=%d seed=%d budget=%d len=%d\n",
		p.MaxTokens, ftoa(p.Temp), ftoa(p.TopP), p.TopK, p.Seed, p.Budget, len(prompt))
	if _, err := io.WriteString(e.in, req+prompt); err != nil {
		return GenResult{}, e.deadErr()
	}

	var res GenResult
	cancelled := false
	cancel := func() {
		if !cancelled {
			cancelled = true
			_, _ = io.WriteString(e.in, "X\n")
		}
	}
	stopWatch := make(chan struct{})
	defer close(stopWatch)
	go func() {
		select {
		case <-ctx.Done():
			e.mu.Lock()
			cancel()
			e.mu.Unlock()
		case <-stopWatch:
		}
	}()
	var cbErr error
	for {
		line, err := e.out.ReadString('\n')
		if err != nil {
			return res, e.deadErr()
		}
		line = strings.TrimRight(line, "\n")
		if line == "" {
			continue
		}
		f := strings.Fields(line)
		switch f[0] {
		case "P":
			kv := fields(f[1:])
			res.PromptTokens, _ = strconv.Atoi(kv["prompt"])
			res.CachedTokens, _ = strconv.Atoi(kv["cached"])
			us, _ := strconv.Atoi(kv["us"])
			res.Prefill = time.Duration(us) * time.Microsecond
			if onPrefill != nil {
				onPrefill(res)
			}
		case "T":
			if len(f) != 4 || cbErr != nil {
				continue // after a stop or an error, drain the rest
			}
			b, err := hex.DecodeString(f[3])
			if err != nil {
				return res, fmt.Errorf("engine: bad token line %q", line)
			}
			if err := onToken(f[2] == "s", b); err != nil {
				cbErr = err
				e.mu.Lock()
				cancel()
				e.mu.Unlock()
			}
		case "E":
			res.Reason = f[1]
			kv := fields(f[2:])
			res.GenTokens, _ = strconv.Atoi(kv["gen"])
			res.ThinkTokens, _ = strconv.Atoi(kv["think"])
			us, _ := strconv.Atoi(kv["us"])
			res.Gen = time.Duration(us) * time.Microsecond
			if errors.Is(cbErr, ErrStop) {
				res.Reason = "stop"
				return res, nil
			}
			if cbErr != nil {
				return res, cbErr
			}
			if res.Reason == "cancel" && ctx.Err() != nil {
				return res, ctx.Err()
			}
			return res, nil
		case "F":
			return res, fmt.Errorf("engine rejected the request: %s", strings.TrimPrefix(line, "F "))
		default:
			return res, fmt.Errorf("engine: unexpected line %q", line)
		}
	}
}

func (e *Engine) deadErr() error {
	select {
	case <-e.dead:
		e.mu.Lock()
		defer e.mu.Unlock()
		return e.err
	case <-time.After(2 * time.Second):
		return errors.New("engine stopped responding")
	}
}

func fields(f []string) map[string]string {
	m := make(map[string]string, len(f))
	for _, kv := range f {
		if k, v, ok := strings.Cut(kv, "="); ok {
			m[k] = v
		}
	}
	return m
}

func ftoa(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }
