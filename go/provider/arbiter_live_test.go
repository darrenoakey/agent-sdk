//go:build live

package provider

// Integration tests against the real Arbiter GPU service (Complete/Stream/Embed
// plus a cold-qwen path with a 900s timeout). Too slow for the default gate.

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func requireArbiterProvider(t *testing.T) *ArbiterProvider {
	t.Helper()
	direct := NewArbiterProvider("")
	if ok, err := direct.Available(context.Background()); err == nil && ok {
		return direct
	}
	return newArbiterTunnelProvider(t)
}

func newArbiterTunnelProvider(t *testing.T) *ArbiterProvider {
	t.Helper()
	port := reserveLoopbackPort(t)
	forward := fmt.Sprintf("127.0.0.1:%d:127.0.0.1:8400", port)
	command := exec.Command("/usr/bin/ssh", "-N", "-o", "BatchMode=yes",
		"-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=10",
		"-L", forward, "darren@10.0.0.254")
	if err := command.Start(); err != nil {
		t.Fatalf("starting owned Arbiter SSH tunnel: %v", err)
	}
	t.Cleanup(func() { stopArbiterTunnel(t, command) })
	provider := NewArbiterProvider(fmt.Sprintf("http://127.0.0.1:%d", port))
	waitForArbiterTunnel(t, provider)
	return provider
}

func reserveLoopbackPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserving loopback port: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("releasing reserved loopback port: %v", err)
	}
	return port
}

func waitForArbiterTunnel(t *testing.T, provider *ArbiterProvider) {
	t.Helper()
	deadline := time.NewTimer(15 * time.Second)
	ticker := time.NewTicker(50 * time.Millisecond)
	t.Cleanup(func() { deadline.Stop() })
	t.Cleanup(ticker.Stop)
	for {
		if ok, err := provider.Available(context.Background()); err == nil && ok {
			return
		}
		select {
		case <-deadline.C:
			t.Fatal("owned Arbiter SSH tunnel did not become ready")
		case <-ticker.C:
		}
	}
}

func stopArbiterTunnel(t *testing.T, command *exec.Cmd) {
	t.Helper()
	if err := command.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Errorf("stopping owned Arbiter SSH tunnel: %v", err)
	}
	if err := command.Wait(); err != nil {
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) {
			t.Errorf("waiting for owned Arbiter SSH tunnel: %v", err)
		}
	}
}

func TestArbiterAvailableReportsReachability(t *testing.T) {
	p := NewArbiterProvider("")
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available returned error: %v", err)
	}
	if ok {
		return
	}
	tunnelProvider := newArbiterTunnelProvider(t)
	ok, err = tunnelProvider.Available(context.Background())
	if err != nil {
		t.Fatalf("Available through owned SSH tunnel returned error: %v", err)
	}
	if !ok {
		t.Fatal("Arbiter unavailable through owned SSH tunnel")
	}
}

func TestArbiterListModels(t *testing.T) {
	p := requireArbiterProvider(t)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("ListModels returned no models")
	}

	names := map[string]bool{}
	for _, m := range models {
		if m.Provider != "arbiter" {
			t.Errorf("model %q has provider %q, want arbiter", m.ModelID, m.Provider)
		}
		if m.ModelID == "" {
			t.Errorf("model has empty ModelID")
		}
		names[m.ModelID] = true
	}
	if !names["nemotron-30b-a3b"] {
		t.Error("ListModels should include nemotron-30b-a3b")
	}
	if !names["ornith-1.5-35b"] {
		t.Error("ListModels should include ornith-1.5-35b")
	}
}

func TestArbiterCompleteSimplePrompt(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "What is 2+2? Reply with just the number."},
	}
	model := testArbiterModel()
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if !strings.Contains(resp.Text, "4") {
		t.Errorf("response text = %q, expected to contain '4'", resp.Text)
	}
	if resp.ModelUsed.ModelID != model.ModelID {
		t.Errorf("ModelUsed = %q, want %q", resp.ModelUsed.ModelID, model.ModelID)
	}
}

func TestArbiterCompleteWithSystemMessage(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "system", Content: "You are a terse assistant. Be brief."},
		{Role: "user", Content: "What is the capital of France?"},
	}
	model := testArbiterModel()
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if !strings.Contains(resp.Text, "Paris") {
		t.Errorf("response text = %q, expected to contain 'Paris'", resp.Text)
	}
}

func TestArbiterStreamYieldsChunks(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "Count from 1 to 5, one number per line."},
	}
	model := testArbiterModel()
	ch, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	var total string
	var chunkCount int
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		if chunk.Text != "" {
			chunkCount++
			total += chunk.Text
		}
	}
	if chunkCount == 0 {
		t.Error("Stream produced no non-empty chunks")
	}
	if strings.TrimSpace(total) == "" {
		t.Error("Stream produced empty full text")
	}
}

func TestArbiterStreamProducesCompleteResponse(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "What is 10 divided by 2? Reply with just the number."},
	}
	model := testArbiterModel()
	ch, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	var full string
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		full += chunk.Text
	}
	if !strings.Contains(full, "5") {
		t.Errorf("stream total = %q, expected to contain '5'", full)
	}
}

func TestArbiterEmbedReturnsVectors(t *testing.T) {
	p := requireArbiterProvider(t)

	texts := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Vector embeddings represent text as points in high-dimensional space.",
		"Nomic embed text v1.5 produces 768-dimensional L2-normalized vectors.",
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*60*1e9)
	defer cancel()
	res, err := p.Embed(ctx, texts, EmbedOpts{Task: "search_document", Timeout: 600})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}
	if res == nil {
		t.Fatal("Embed returned nil result")
	}
	if len(res.Embeddings) != len(texts) {
		t.Fatalf("got %d embeddings, want %d", len(res.Embeddings), len(texts))
	}
	for i, vec := range res.Embeddings {
		if len(vec) == 0 {
			t.Fatalf("embedding %d is empty", i)
		}
	}
	if res.Task != "search_document" {
		t.Errorf("Task = %q, want search_document", res.Task)
	}
	if res.ModelUsed.Provider != "arbiter" {
		t.Errorf("ModelUsed.Provider = %q, want arbiter", res.ModelUsed.Provider)
	}
}

func TestArbiterEmbedDimension(t *testing.T) {
	p := requireArbiterProvider(t)

	ctx, cancel := context.WithTimeout(context.Background(), 10*60*1e9)
	defer cancel()
	res, err := p.Embed(ctx, []string{"dimension check"}, EmbedOpts{Timeout: 600})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}
	if len(res.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(res.Embeddings))
	}
	if res.Dimension != 768 {
		t.Errorf("Dimension = %d, want 768", res.Dimension)
	}
	if len(res.Embeddings[0]) != 768 {
		t.Errorf("len(Embeddings[0]) = %d, want 768", len(res.Embeddings[0]))
	}
	if res.Dimension != len(res.Embeddings[0]) {
		t.Errorf("Dimension %d != len(Embeddings[0]) %d", res.Dimension, len(res.Embeddings[0]))
	}
}

func TestArbiterCompleteQwenReasoning(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "What is 2+2? Answer with just the number."},
	}
	model := sdk.ModelInfo{
		Provider:             "arbiter",
		ModelID:              "local-coder",
		DisplayName:          "Local Coder",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierFreeThinking,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
	}
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 900})
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if strings.TrimSpace(resp.Text) == "" {
		t.Error("qwen response text is empty — reasoning fallback did not populate it")
	}
}

func TestArbiterLoopbackTunnelListsModels(t *testing.T) {
	provider := newArbiterTunnelProvider(t)
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels through owned SSH tunnel: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("ListModels through owned SSH tunnel returned no LLM models")
	}
}
