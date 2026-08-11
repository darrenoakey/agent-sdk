//go:build live

package dazagentsdk

import (
	"context"
	"testing"
	"time"
)

// TestAgent_AskOllama tests Ask against a real Ollama instance.
func TestAgent_AskOllama(t *testing.T) {
	baseURL := configuredTestOllamaURL(t)
	RegisterProviderFactory("ollama", func(_ *Config) Provider {
		return newTestOllamaProvider(baseURL)
	})
	defer RefreshProviders()

	agent := NewAgent(nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	resp, err := agent.Ask(ctx, "Say hello in exactly one word.",
		WithAskTier(TierFreeFast),
		WithAskProvider("ollama"),
		WithAskModel("llama3.2:3b"),
	)
	if err != nil {
		t.Fatalf("Ask() error: %v", err)
	}

	if resp.Text == "" {
		t.Error("Ask() returned empty text")
	}
}
