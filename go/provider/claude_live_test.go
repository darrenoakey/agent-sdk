//go:build live

package provider

import (
	"context"
	"strings"
	"testing"
	"time"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func requireClaudeCLI(t *testing.T) {
	t.Helper()
	if _, err := findClaudeCLI(); err != nil {
		t.Fatalf("Claude CLI unavailable: %v", err)
	}
}

func TestClaudeComplete_BasicText(t *testing.T) {
	requireClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "What is 2+2? Reply with just the number."}}
	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !strings.Contains(resp.Text, "4") {
		t.Errorf("expected '4' in response, got: %s", resp.Text)
	}
}

func TestClaudeComplete_SystemMessage(t *testing.T) {
	requireClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{
		{Role: "system", Content: "Always respond in exactly one word."},
		{Role: "user", Content: "Say hello."},
	}
	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if strings.TrimSpace(resp.Text) == "" {
		t.Error("expected non-empty response")
	}
}

func TestClaudeComplete_StructuredOutput(t *testing.T) {
	requireClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "What is 10 + 5?"}}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "integer"},
		},
		"required": []string{"answer"},
	}

	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{
		Schema:  schema,
		Timeout: 60,
	})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !strings.Contains(resp.Text, "15") {
		t.Errorf("structured response = %q, want answer 15", resp.Text)
	}
}

func TestClaudeStream_BasicText(t *testing.T) {
	requireClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "Say hello in one word."}}
	ch, err := p.Stream(context.Background(), messages, haikuModel(), sdk.StreamOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	var text strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		text.WriteString(chunk.Text)
	}
	if text.Len() == 0 {
		t.Error("expected non-empty streamed text")
	}
}

func TestClaudeComplete_ModelUsed(t *testing.T) {
	requireClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "Say hi."}}
	resp, err := p.Complete(context.Background(), messages, haikuModel(), sdk.CompleteOpts{Timeout: 30})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if resp.ModelUsed.ModelID != "claude-haiku-4-5-20251001" {
		t.Errorf("ModelUsed.ModelID = %q, want claude-haiku-4-5-20251001", resp.ModelUsed.ModelID)
	}
	if resp.ConversationID.String() == "00000000-0000-0000-0000-000000000000" {
		t.Error("ConversationID should be non-zero")
	}
}

func TestClaudeStream_Timeout(t *testing.T) {
	requireClaudeCLI(t)

	p := NewClaudeProvider()
	messages := []sdk.Message{{Role: "user", Content: "Write a very long essay."}}
	ch, err := p.Stream(context.Background(), messages, haikuModel(), sdk.StreamOpts{Timeout: 0.001})
	if err != nil {
		return
	}
	timeout := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-ch:
			if !ok {
				return
			}
		case <-timeout:
			t.Fatal("stream did not terminate within timeout")
		}
	}
}
