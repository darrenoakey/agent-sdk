package provider

import (
	"context"
	"testing"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func testArbiterModel() sdk.ModelInfo {
	return sdk.ModelInfo{
		Provider:             "arbiter",
		ModelID:              "local-coder",
		DisplayName:          "Test Arbiter Model",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierFreeThinking,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        false,
	}
}

func TestKnownArbiterTiers(t *testing.T) {
	if knownArbiterTiers["local-chat"] != sdk.TierFreeFast {
		t.Errorf("local-chat should be FreeFast")
	}
	if knownArbiterTiers["local-summariser"] != sdk.TierSummaries {
		t.Errorf("local-summariser should be Summaries")
	}
	if knownArbiterTiers["local-coder"] != sdk.TierFreeThinking {
		t.Errorf("local-coder should be FreeThinking")
	}
	if knownArbiterTiers["local-extract"] != sdk.TierFreeThinking {
		t.Errorf("local-extract should be FreeThinking")
	}
	if knownArbiterTiers["local-vision"] != sdk.TierFreeThinking {
		t.Errorf("local-vision should be FreeThinking")
	}
	if knownArbiterTiers["qwen3.8-27b"] != sdk.TierFreeThinking {
		t.Errorf("qwen3.8-27b should map to FreeThinking")
	}
	if knownArbiterTiers["qwen3.6-35b"] != sdk.TierFreeThinking {
		t.Errorf("qwen3.6-35b should still map to FreeThinking (back-compat)")
	}
	if knownArbiterTiers["nemotron-30b-a3b"] != sdk.TierFreeFast {
		t.Errorf("nemotron-30b-a3b should be FreeFast")
	}
	if knownArbiterTiers["ornith-1.5-35b"] != sdk.TierFreeThinking {
		t.Errorf("ornith-1.5-35b should be FreeThinking")
	}
}

func TestNewArbiterProviderDefaults(t *testing.T) {
	p := NewArbiterProvider("")
	if p.Name() != "arbiter" {
		t.Errorf("Name() = %q, want arbiter", p.Name())
	}
	if p.baseURL != "http://10.0.0.254:8400" {
		t.Errorf("baseURL = %q, want http://10.0.0.254:8400", p.baseURL)
	}
}

func TestNewArbiterProviderCustomURL(t *testing.T) {
	p := NewArbiterProvider("http://myhost:9999/")
	if p.baseURL != "http://myhost:9999" {
		t.Errorf("baseURL = %q, want http://myhost:9999", p.baseURL)
	}
}

func TestArbiterAvailableWrongPortReturnsFalse(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available returned error on unreachable: %v", err)
	}
	if ok {
		t.Error("Available should return false for unreachable arbiter")
	}
}

func TestArbiterListModelsWrongPortReturnsError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	_, err := p.ListModels(context.Background())
	if err == nil {
		t.Error("ListModels should error when arbiter is unreachable")
	}
}

func TestArbiterCompleteWrongPortRaisesAgentError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	messages := []sdk.Message{{Role: "user", Content: "hello"}}
	model := testArbiterModel()
	_, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 5})
	if err == nil {
		t.Fatal("Complete should error against unreachable arbiter")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorNotAvailable {
		t.Errorf("Kind = %v, want ErrorNotAvailable", agentErr.Kind)
	}
}

func TestArbiterStreamWrongPortRaisesAgentError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	messages := []sdk.Message{{Role: "user", Content: "hello"}}
	model := testArbiterModel()
	_, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{Timeout: 5})
	if err == nil {
		t.Fatal("Stream should error against unreachable arbiter")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorNotAvailable {
		t.Errorf("Kind = %v, want ErrorNotAvailable", agentErr.Kind)
	}
}

func TestArbiterEmbedEmptyTextsRaisesError(t *testing.T) {
	p := NewArbiterProvider("")
	_, err := p.Embed(context.Background(), nil, EmbedOpts{})
	if err == nil {
		t.Fatal("Embed([]) should error")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorInvalidRequest {
		t.Errorf("Kind = %v, want ErrorInvalidRequest", agentErr.Kind)
	}
}

func TestArbiterEmbedWrongPortRaisesAgentError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	_, err := p.Embed(context.Background(), []string{"hi"}, EmbedOpts{Timeout: 5})
	if err == nil {
		t.Fatal("Embed should error against unreachable arbiter")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorNotAvailable {
		t.Errorf("Kind = %v, want ErrorNotAvailable", agentErr.Kind)
	}
}
