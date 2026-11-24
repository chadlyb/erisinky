// Command inkserve serves Inkling 2 Mini through an OpenAI-style HTTP API, using erisinky's Metal engine
// (`inkling serve`) as a child process. See README.md.
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

func main() {
	home, _ := os.UserHomeDir()
	listen := flag.String("listen", "127.0.0.1:8045", "address to serve the API on")
	name := flag.String("name", "inkling-erisinky", "model id the API reports and accepts")
	engineBin := flag.String("engine", "~/erisinky-eval/src/target/release/inkling", "erisinky binary with the serve command")
	model := flag.String("model", "/Volumes/TB5/llm/inkling-speed/Inkling-2-Mini-Q4_K_M.gguf", "Inkling 2 Mini Q4_K_M GGUF")
	ctx := flag.Int("ctx", 131072, "context length in tokens")
	effort := flag.String("effort", "high", "default reasoning effort: none, minimal, low, medium, high, xhigh, or 0–0.99")
	budget := flag.Int("budget", 8192, "default thinking budget in tokens (0: none)")
	temp := flag.Float64("temperature", 1.0, "default temperature")
	topP := flag.Float64("top-p", 1.0, "default top-p")
	topK := flag.Int("top-k", 0, "default top-k (0: off)")
	maxTokens := flag.Int("max-tokens", 0, "default reply limit in tokens (0: up to the end of the context)")
	dspark := flag.String("dspark", "", "DSpark drafter GGUF: speculative decoding, a few percent faster on agent replies (off when empty)")
	flag.Parse()

	logger := log.New(os.Stderr, "inkserve: ", log.LstdFlags)
	eff, err := ParseEffort(*effort)
	if err != nil {
		logger.Fatalf("-effort: %v", err)
	}
	expand := func(p string) string {
		if strings.HasPrefix(p, "~/") {
			return filepath.Join(home, p[2:])
		}
		return p
	}

	logger.Printf("loading %s (context %d)", *model, *ctx)
	drafter := ""
	if *dspark != "" {
		drafter = expand(*dspark)
		logger.Printf("DSpark on, drafter %s", drafter)
	}
	eng, err := StartEngine(expand(*engineBin), expand(*model), drafter, *ctx, os.Stderr)
	if err != nil {
		logger.Fatal(err)
	}
	logger.Printf("engine ready: context %d, vocabulary %d", eng.Ctx, eng.Vocab)

	srv := &Server{Model: *name, Engine: eng, Log: logger, Defaults: Defaults{
		Effort: eff, Budget: *budget, Temperature: *temp, TopP: *topP, TopK: *topK, MaxTokens: *maxTokens}}
	hs := &http.Server{Addr: *listen, Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	engineDied := make(chan struct{})
	go func() {
		select {
		case s := <-sig:
			logger.Printf("%v: shutting down", s)
		case <-eng.Dead():
			logger.Printf("%v", eng.deadErr())
			close(engineDied)
		}
		c, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = hs.Shutdown(c)
	}()
	logger.Printf("serving %s on http://%s", *name, *listen)
	if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Fatal(err)
	}
	select {
	case <-engineDied:
		os.Exit(1) // a supervisor (m5ctl, launchd) can restart us
	default:
		_ = eng.Close()
	}
}
