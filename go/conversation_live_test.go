//go:build live

package dazagentsdk

import (
	"context"
	"testing"
	"time"
)

// TestConversation_SayOllama tests Say against a real Ollama instance.
func TestConversation_SayOllama(t *testing.T) {
	baseURL := configuredTestOllamaURL(t)
	RegisterProviderFactory("ollama", func(_ *Config) Provider {
		return newTestOllamaProvider(baseURL)
	})
	defer RefreshProviders()

	conv := NewConversation("ollama-test",
		WithTier(TierFreeFast),
		WithProvider("ollama"),
		WithModel("llama3.2:3b"),
	)
	defer conv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	resp, err := conv.Say(ctx, "Say hello in exactly one word.")
	if err != nil {
		t.Fatalf("Say() error: %v", err)
	}

	if resp.Text == "" {
		t.Error("Say() returned empty text")
	}

	hist := conv.History()
	if len(hist) < 2 {
		t.Fatalf("History() len = %d, want >= 2 (user + assistant)", len(hist))
	}
	if hist[len(hist)-2].Role != "user" {
		t.Errorf("second-to-last message role = %q, want %q", hist[len(hist)-2].Role, "user")
	}
	if hist[len(hist)-1].Role != "assistant" {
		t.Errorf("last message role = %q, want %q", hist[len(hist)-1].Role, "assistant")
	}
}
