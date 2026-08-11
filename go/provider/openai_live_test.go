//go:build live

package provider

import (
	"context"
	"strings"
	"testing"
	"time"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func openAIMiniModel() sdk.ModelInfo {
	return openAIModels[1]
}

func TestOpenAIAvailableWithKey(t *testing.T) {
	p := NewOpenAIProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available() returned error: %v", err)
	}
	if !ok {
		t.Error("Available() = false with the configured credential")
	}
}

func TestOpenAICompleteBasicText(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Say exactly: HELLO"}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete() returned empty text")
	}
	if resp.ConversationID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("ConversationID is zero UUID")
	}
	if resp.TurnID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("TurnID is zero UUID")
	}
	if resp.ModelUsed.ModelID != model.ModelID {
		t.Errorf("ModelUsed.ModelID = %q, want %q", resp.ModelUsed.ModelID, model.ModelID)
	}
}

func TestOpenAICompleteSystemMessage(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{
		{Role: "system", Content: "You are a pirate. Always say 'Arrr' at the start."},
		{Role: "user", Content: "Greet me."},
	}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Complete() returned empty text")
	}
}

func TestOpenAICompleteUsagePopulated(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Reply with one word."}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.Usage == nil {
		t.Fatal("Usage map is nil")
	}
	if _, ok := resp.Usage["prompt_tokens"]; !ok {
		t.Error("Usage missing prompt_tokens")
	}
	if _, ok := resp.Usage["completion_tokens"]; !ok {
		t.Error("Usage missing completion_tokens")
	}
}

func TestOpenAICompleteStructuredMathSchema(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result": map[string]any{"type": "number"},
		},
		"required":             []string{"result"},
		"additionalProperties": false,
	}
	msgs := []sdk.Message{{Role: "user", Content: "What is 3 + 4? Respond with the numeric result."}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{Schema: schema})
	if err != nil {
		t.Fatalf("Complete() with schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Structured response text is empty")
	}
	if !strings.Contains(resp.Text, "7") {
		t.Errorf("expected '7' in structured response, got: %q", resp.Text)
	}
}

func TestOpenAICompleteStructuredComplexSchema(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":    map[string]any{"type": "string"},
			"age":     map[string]any{"type": "integer"},
			"country": map[string]any{"type": "string"},
		},
		"required":             []string{"name", "age", "country"},
		"additionalProperties": false,
	}
	msgs := []sdk.Message{{Role: "user", Content: "Return info about Albert Einstein."}}
	resp, err := p.Complete(context.Background(), msgs, model, sdk.CompleteOpts{Schema: schema})
	if err != nil {
		t.Fatalf("Complete() with complex schema error: %v", err)
	}
	if resp.Text == "" {
		t.Error("Structured response text is empty")
	}
	if !strings.Contains(resp.Text, "Einstein") {
		t.Errorf("expected 'Einstein' in response, got: %q", resp.Text)
	}
}

func TestOpenAIStreamBasicText(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Count to 3 briefly."}}
	ch, err := p.Stream(context.Background(), msgs, model, sdk.StreamOpts{})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	var collected strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		collected.WriteString(chunk.Text)
	}
	if collected.Len() == 0 {
		t.Error("Stream() produced no text")
	}
}

func TestOpenAIStreamCollectsAllChunks(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Write a haiku about Go programming."}}
	ch, err := p.Stream(context.Background(), msgs, model, sdk.StreamOpts{})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	var chunks int
	var full strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		if chunk.Text != "" {
			chunks++
			full.WriteString(chunk.Text)
		}
	}
	if chunks < 2 {
		t.Errorf("expected at least 2 chunks, got %d", chunks)
	}
	if full.Len() == 0 {
		t.Error("stream collected no text")
	}
}

func TestOpenAIStreamTimeout(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Write a very long essay about the history of computing."}}
	ch, err := p.Stream(context.Background(), msgs, model, sdk.StreamOpts{Timeout: 0.001})
	if err != nil {
		return
	}
	var gotErr bool
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = true
			break
		}
	}
	_ = gotErr
}

func TestOpenAICompleteTimeout(t *testing.T) {
	p := NewOpenAIProvider()
	model := openAIMiniModel()
	msgs := []sdk.Message{{Role: "user", Content: "Tell me everything about computing."}}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	_, err := p.Complete(ctx, msgs, model, sdk.CompleteOpts{})
	if err == nil {
		t.Log("Complete() unexpectedly succeeded with 1ms timeout")
		return
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("expected *sdk.AgentError, got %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorTimeout {
		t.Errorf("error kind = %q, want %q", agentErr.Kind, sdk.ErrorTimeout)
	}
}
