//go:build live

package provider

import (
	"context"
	"encoding/json"
	"testing"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// Integration tests against the real Ollama instance at localhost:11434.

func requireOllamaAvailable(t *testing.T, p *OllamaProvider) {
	t.Helper()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("checking Ollama availability: %v", err)
	}
	if !ok {
		t.Fatal("Ollama is not available at localhost:11434")
	}
}

func TestOllamaAvailable(t *testing.T) {
	p := NewOllamaProvider("")
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available returned error: %v", err)
	}
	// We don't fail if Ollama is down -- just report
	t.Logf("Ollama available: %v", ok)
}

func TestOllamaListModels(t *testing.T) {
	p := NewOllamaProvider("")
	requireOllamaAvailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("no models installed in Ollama")
	}

	for _, m := range models {
		if m.Provider != "ollama" {
			t.Errorf("model %q has provider %q, want ollama", m.ModelID, m.Provider)
		}
		if m.ModelID == "" {
			t.Error("model has empty ModelID")
		}
		if m.DisplayName == "" {
			t.Error("model has empty DisplayName")
		}
		if len(m.Capabilities) == 0 {
			t.Errorf("model %q has no capabilities", m.ModelID)
		}
		// Chat models expose text/structured; embedding-only models do not.
		hasText := false
		hasStructured := false
		hasEmbedding := false
		for _, c := range m.Capabilities {
			if c == sdk.CapabilityText {
				hasText = true
			}
			if c == sdk.CapabilityStructured {
				hasStructured = true
			}
			if c == sdk.CapabilityEmbedding {
				hasEmbedding = true
			}
		}
		if hasEmbedding && (hasText || hasStructured || m.SupportsStreaming) {
			t.Errorf("embedding model %q advertises chat support", m.ModelID)
		}
		if !hasEmbedding && !hasText {
			t.Errorf("model %q missing text capability", m.ModelID)
		}
		if !hasEmbedding && !hasStructured {
			t.Errorf("model %q missing structured capability", m.ModelID)
		}
	}

	t.Logf("Found %d Ollama models", len(models))
}

func TestOllamaComplete(t *testing.T) {
	p := NewOllamaProvider("")
	requireOllamaAvailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("no models available for Complete test")
	}

	// Use the first available model
	model := models[0]
	messages := []sdk.Message{
		{Role: "user", Content: "Say exactly: hello world"},
	}

	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}
	if resp.ModelUsed.ModelID != model.ModelID {
		t.Errorf("ModelUsed.ModelID = %q, want %q", resp.ModelUsed.ModelID, model.ModelID)
	}
	t.Logf("Complete response: %q", resp.Text)
}

func TestOllamaStream(t *testing.T) {
	p := NewOllamaProvider("")
	requireOllamaAvailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("no models available for Stream test")
	}

	model := models[0]
	messages := []sdk.Message{
		{Role: "user", Content: "Say exactly: hello"},
	}

	ch, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var fullText string
	chunks := 0
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("Stream chunk error: %v", chunk.Err)
		}
		fullText += chunk.Text
		chunks++
	}

	if fullText == "" {
		t.Error("Stream produced no text")
	}
	t.Logf("Stream produced %d chunks, full text: %q", chunks, fullText)
}

// firstModelWithStructured returns the first model that advertises structured
// output support, or the first model overall if none advertise it.
func firstModelWithStructured(models []sdk.ModelInfo) sdk.ModelInfo {
	for _, m := range models {
		if m.SupportsStructured {
			return m
		}
	}
	return models[0]
}

// TestOllamaCompleteStructuredSimple is an integration test that asks a
// simple maths question and expects a JSON response matching a basic schema.
func TestOllamaCompleteStructuredSimple(t *testing.T) {
	p := NewOllamaProvider("")
	requireOllamaAvailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("no models available")
	}
	model := firstModelWithStructured(models)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "number"},
		},
		"required": []string{"answer"},
	}

	resp, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "system", Content: "Respond only with valid JSON matching the given schema."},
		{Role: "user", Content: "What is 6 multiplied by 7? Respond as JSON with key 'answer'."},
	}, model, sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete structured error: %v", err)
	}
	if resp.Text == "" {
		t.Fatal("structured response text is empty")
	}
	t.Logf("Structured response: %q", resp.Text)

	// The response should be parseable JSON
	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Errorf("response is not valid JSON: %v", err)
	}
}

// TestOllamaCompleteStructuredComplex is an integration test using a
// multi-field schema to retrieve a structured person description.
func TestOllamaCompleteStructuredComplex(t *testing.T) {
	p := NewOllamaProvider("")
	requireOllamaAvailable(t, p)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("no models available")
	}
	model := firstModelWithStructured(models)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":       map[string]any{"type": "string"},
			"occupation": map[string]any{"type": "string"},
			"age":        map[string]any{"type": "integer"},
		},
		"required": []string{"name", "occupation", "age"},
	}

	resp, err := p.Complete(context.Background(), []sdk.Message{
		{Role: "system", Content: "Respond only with valid JSON matching the given schema."},
		{Role: "user", Content: "Invent a fictional person and describe them as JSON with keys: name, occupation, age."},
	}, model, sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete structured complex error: %v", err)
	}
	if resp.Text == "" {
		t.Fatal("structured response text is empty")
	}
	t.Logf("Complex structured response: %q", resp.Text)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Errorf("response is not valid JSON: %v", err)
	}
	for _, key := range []string{"name", "occupation", "age"} {
		if _, ok := parsed[key]; !ok {
			t.Errorf("response JSON missing required key %q", key)
		}
	}
}
