//go:build live

package provider

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func requireGeminiCredential(t *testing.T) {
	t.Helper()
	if _, err := geminiAPIKey(context.Background()); err != nil {
		t.Fatalf("Gemini credential unavailable: %v", err)
	}
}

func geminiFlashLite() sdk.ModelInfo {
	return sdk.ModelInfo{
		Provider: "gemini",
		ModelID:  "gemini-3.1-flash-lite",
	}
}

func TestGeminiAvailable_WithKey(t *testing.T) {
	requireGeminiCredential(t)
	p := NewGeminiProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available error: %v", err)
	}
	if !ok {
		t.Error("expected Available to return true with valid API key")
	}
}

func TestGeminiComplete_BasicText(t *testing.T) {
	requireGeminiCredential(t)
	p := NewGeminiProvider()
	messages := []sdk.Message{{Role: "user", Content: "Reply with exactly one word: pong"}}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{Timeout: 60})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}
	if resp.ModelUsed.ModelID != "gemini-3.1-flash-lite" {
		t.Errorf("ModelUsed.ModelID = %q, want gemini-3.1-flash-lite", resp.ModelUsed.ModelID)
	}
	if resp.ConversationID.String() == "" {
		t.Error("ConversationID should be set")
	}
	if resp.TurnID.String() == "" {
		t.Error("TurnID should be set")
	}
	t.Logf("Complete response: %q", resp.Text)
}

func TestGeminiComplete_StructuredOutputSimple(t *testing.T) {
	requireGeminiCredential(t)
	p := NewGeminiProvider()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result": map[string]any{
				"type":        "integer",
				"description": "The numeric answer",
			},
		},
		"required": []any{"result"},
	}
	messages := []sdk.Message{{Role: "user", Content: "What is 3 + 4? Respond in JSON with a 'result' field."}}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{Schema: schema, Timeout: 60})
	if err != nil {
		t.Fatalf("Complete with schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Fatalf("structured output is not valid JSON: %v, text: %q", err, resp.Text)
	}
	result, ok := parsed["result"]
	if !ok {
		t.Errorf("parsed JSON missing 'result' field, got: %v", parsed)
	}
	t.Logf("Structured result: %v", result)
}

func TestGeminiComplete_StructuredOutputComplex(t *testing.T) {
	requireGeminiCredential(t)
	p := NewGeminiProvider()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": "The name of the capital city",
			},
			"country": map[string]any{
				"type":        "string",
				"description": "The country name",
			},
			"population_millions": map[string]any{
				"type":        "number",
				"description": "Approximate population in millions",
			},
		},
		"required": []any{"name", "country", "population_millions"},
	}
	messages := []sdk.Message{{Role: "user", Content: "Give me information about France's capital city in JSON format."}}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{Schema: schema, Timeout: 60})
	if err != nil {
		t.Fatalf("Complete with complex schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete returned empty text")
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &parsed); err != nil {
		t.Fatalf("structured output is not valid JSON: %v, text: %q", err, resp.Text)
	}
	for _, field := range []string{"name", "country", "population_millions"} {
		if _, ok := parsed[field]; !ok {
			t.Errorf("parsed JSON missing required field %q, got keys: %v", field, keys(parsed))
		}
	}
	t.Logf("Complex structured result: %v", parsed)
}

func TestGeminiStream_BasicText(t *testing.T) {
	requireGeminiCredential(t)
	p := NewGeminiProvider()
	messages := []sdk.Message{{Role: "user", Content: "Reply with exactly one word: ping"}}
	ch, err := p.Stream(context.Background(), messages, geminiFlashLite(), sdk.StreamOpts{Timeout: 60})
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

func TestGeminiComplete_SystemMessage(t *testing.T) {
	requireGeminiCredential(t)
	p := NewGeminiProvider()
	messages := []sdk.Message{
		{Role: "system", Content: "You only ever reply with the single word MANGO."},
		{Role: "user", Content: "What is your favourite fruit?"},
	}
	resp, err := p.Complete(context.Background(), messages, geminiFlashLite(), sdk.CompleteOpts{Timeout: 60})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !strings.Contains(strings.ToUpper(resp.Text), "MANGO") {
		t.Errorf("expected MANGO in response, got: %q", resp.Text)
	}
	t.Logf("System-prompted response: %q", resp.Text)
}

func keys(m map[string]any) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}
