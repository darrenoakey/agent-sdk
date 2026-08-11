package provider

import (
	"context"
	"fmt"
	"os/exec"
	"slices"
	"strings"
	"testing"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// ── Claude CLI discovery ─────────────────────────────────────────

func TestFindClaudeCLI(t *testing.T) {
	path, err := findClaudeCLI()
	if err != nil {
		if path != "" {
			t.Errorf("findClaudeCLI returned path %q with error %v", path, err)
		}
		return
	}
	if path == "" {
		t.Error("findClaudeCLI returned empty path")
	}
	t.Logf("found claude CLI at: %s", path)
}

func TestFindClaudeCLI_MatchesWhich(t *testing.T) {
	path, err := findClaudeCLI()
	if err != nil {
		if _, lookErr := exec.LookPath("claude"); lookErr == nil {
			t.Fatalf("findClaudeCLI failed despite claude being on PATH: %v", err)
		}
		return
	}
	whichPath, err := exec.LookPath("claude")
	if err != nil {
		t.Logf("claude found through a production fallback path: %s", path)
		return
	}
	if path != whichPath {
		t.Logf("findClaudeCLI=%s, which=%s (may differ if found in fallback path)", path, whichPath)
	}
}

// ── Provider basics ──────────────────────────────────────────────

func TestClaudeName(t *testing.T) {
	p := NewClaudeProvider()
	if p.Name() != "claude" {
		t.Errorf("Name() = %q, want claude", p.Name())
	}
}

func TestClaudeAvailable(t *testing.T) {
	p := NewClaudeProvider()
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available error: %v", err)
	}
	_, cliErr := findClaudeCLI()
	expected := cliErr == nil
	if ok != expected {
		t.Errorf("Available() = %v, want %v", ok, expected)
	}
}

func TestClaudeListModels(t *testing.T) {
	p := NewClaudeProvider()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) != 3 {
		t.Errorf("len(models) = %d, want 3", len(models))
	}
	ids := map[string]bool{}
	for _, m := range models {
		ids[m.ModelID] = true
		if m.Provider != "claude" {
			t.Errorf("model %s has provider %q, want claude", m.ModelID, m.Provider)
		}
	}
	for _, expected := range []string{"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"} {
		if !ids[expected] {
			t.Errorf("missing model %s", expected)
		}
	}
}

func TestClaudeListModels_IsCopy(t *testing.T) {
	p := NewClaudeProvider()
	m1, _ := p.ListModels(context.Background())
	m2, _ := p.ListModels(context.Background())
	m1[0].DisplayName = "MUTATED"
	if m2[0].DisplayName == "MUTATED" {
		t.Error("ListModels should return a copy, not a reference to the internal slice")
	}
}

// ── Model catalog ────────────────────────────────────────────────

func TestClaudeModelTiers(t *testing.T) {
	p := NewClaudeProvider()
	models, _ := p.ListModels(context.Background())
	tierMap := map[string]sdk.Tier{}
	for _, m := range models {
		tierMap[m.ModelID] = m.Tier
	}
	if tierMap["claude-opus-4-6"] != sdk.TierHigh {
		t.Errorf("opus tier = %s, want high", tierMap["claude-opus-4-6"])
	}
	if tierMap["claude-sonnet-4-6"] != sdk.TierMedium {
		t.Errorf("sonnet tier = %s, want medium", tierMap["claude-sonnet-4-6"])
	}
	if tierMap["claude-haiku-4-5-20251001"] != sdk.TierLow {
		t.Errorf("haiku tier = %s, want low", tierMap["claude-haiku-4-5-20251001"])
	}
}

func TestClaudeModelCapabilities(t *testing.T) {
	p := NewClaudeProvider()
	models, _ := p.ListModels(context.Background())
	for _, m := range models {
		if !m.SupportsStreaming {
			t.Errorf("model %s should support streaming", m.ModelID)
		}
		if !m.SupportsStructured {
			t.Errorf("model %s should support structured output", m.ModelID)
		}
		if !m.SupportsConversation {
			t.Errorf("model %s should support conversation", m.ModelID)
		}
		if !m.SupportsTools {
			t.Errorf("model %s should support tools", m.ModelID)
		}
	}
}

// ── Prompt building ──────────────────────────────────────────────

func TestBuildPrompt_Simple(t *testing.T) {
	messages := []sdk.Message{{Role: "user", Content: "hello"}}
	system, prompt := buildPrompt(messages)
	if system != "" {
		t.Errorf("system = %q, want empty", system)
	}
	if prompt != "hello" {
		t.Errorf("prompt = %q, want hello", prompt)
	}
}

func TestBuildPrompt_WithSystem(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "you are helpful"},
		{Role: "user", Content: "hi"},
	}
	system, prompt := buildPrompt(messages)
	if system != "you are helpful" {
		t.Errorf("system = %q, want 'you are helpful'", system)
	}
	if prompt != "hi" {
		t.Errorf("prompt = %q, want 'hi'", prompt)
	}
}

func TestBuildPrompt_MultipleSystems(t *testing.T) {
	messages := []sdk.Message{
		{Role: "system", Content: "rule 1"},
		{Role: "system", Content: "rule 2"},
		{Role: "user", Content: "hello"},
	}
	system, _ := buildPrompt(messages)
	if system != "rule 1\nrule 2" {
		t.Errorf("system = %q, want 'rule 1\\nrule 2'", system)
	}
}

func TestBuildPrompt_MultiTurn(t *testing.T) {
	messages := []sdk.Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "response"},
		{Role: "user", Content: "second"},
	}
	_, prompt := buildPrompt(messages)
	if !strings.Contains(prompt, "first") || !strings.Contains(prompt, "response") || !strings.Contains(prompt, "second") {
		t.Errorf("prompt missing expected parts: %q", prompt)
	}
	if !strings.Contains(prompt, "[Previous assistant response]") {
		t.Errorf("prompt should prefix assistant messages: %q", prompt)
	}
}

func TestBuildPrompt_NoMessages(t *testing.T) {
	system, prompt := buildPrompt(nil)
	if system != "" || prompt != "" {
		t.Errorf("empty messages should produce empty strings, got system=%q prompt=%q", system, prompt)
	}
}

// ── Error classification ─────────────────────────────────────────

func TestClassifyClaudeError_RateLimit(t *testing.T) {
	for _, msg := range []string{"rate_limit exceeded", "HTTP 429", "server overloaded"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorRateLimit {
			t.Errorf("classifyClaudeError(%q) = %s, want rate_limit", msg, got)
		}
	}
}

func TestClassifyClaudeError_Auth(t *testing.T) {
	for _, msg := range []string{"401 Unauthorized", "403 Forbidden", "auth failed"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorAuth {
			t.Errorf("classifyClaudeError(%q) = %s, want auth", msg, got)
		}
	}
}

func TestClassifyClaudeError_Timeout(t *testing.T) {
	for _, msg := range []string{"request timed out", "timeout exceeded"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorTimeout {
			t.Errorf("classifyClaudeError(%q) = %s, want timeout", msg, got)
		}
	}
}

func TestClassifyClaudeError_InvalidRequest(t *testing.T) {
	for _, msg := range []string{"400 bad request invalid", "invalid parameters"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorInvalidRequest {
			t.Errorf("classifyClaudeError(%q) = %s, want invalid_request", msg, got)
		}
	}
}

func TestClassifyClaudeError_NotAvailable(t *testing.T) {
	for _, msg := range []string{"claude not found", "CLI not installed"} {
		if got := classifyClaudeError(fmt.Errorf("%s", msg)); got != sdk.ErrorNotAvailable {
			t.Errorf("classifyClaudeError(%q) = %s, want not_available", msg, got)
		}
	}
}

func TestClassifyClaudeError_Internal(t *testing.T) {
	if got := classifyClaudeError(fmt.Errorf("something weird")); got != sdk.ErrorInternal {
		t.Errorf("classifyClaudeError('something weird') = %s, want internal", got)
	}
}

// ── CLAUDECODE env stripping ─────────────────────────────────────

func TestStripClaudeCodeEnv(t *testing.T) {
	env := stripClaudeCodeEnv()
	for _, e := range env {
		if strings.HasPrefix(e, "CLAUDECODE=") {
			t.Error("stripClaudeCodeEnv should remove CLAUDECODE")
		}
	}
}

// ── Integration tests against the installed Claude CLI ──────────

func TestCompleteCommandArgsNativeToolsConfiguration(t *testing.T) {
	tests := []struct {
		name     string
		provider *ClaudeProvider
		expected []string
	}{
		{
			name:     "Default",
			provider: NewClaudeProvider(),
			expected: []string{"--output-format", "stream-json", "--verbose", "--system-prompt", "system", "--model", "claude-haiku-4-5-20251001", "--permission-mode", "bypassPermissions", "--max-turns", "1", "-p", "hello"},
		},
		{
			name:     "NativeToolsDisabled",
			provider: NewClaudeProviderWithOptions(WithClaudeNativeToolsDisabled()),
			expected: []string{"--output-format", "stream-json", "--verbose", "--system-prompt", "system", "--model", "claude-haiku-4-5-20251001", "--permission-mode", "bypassPermissions", "--tools", "", "--max-turns", "1", "-p", "hello"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			arguments, err := test.provider.completeCommandArgs("system", "hello", haikuModel(), sdk.CompleteOpts{})
			if err != nil {
				t.Fatalf("building Complete arguments: %v", err)
			}
			assertStringSlicesEqual(t, arguments, test.expected)
		})
	}
}

func TestStreamCommandArgsNativeToolsConfiguration(t *testing.T) {
	tests := []struct {
		name     string
		provider *ClaudeProvider
		expected []string
	}{
		{
			name:     "Default",
			provider: NewClaudeProvider(),
			expected: []string{"--output-format", "stream-json", "--verbose", "--system-prompt", "system", "--model", "claude-haiku-4-5-20251001", "--permission-mode", "bypassPermissions", "--max-turns", "1", "-p", "hello"},
		},
		{
			name:     "NativeToolsDisabled",
			provider: NewClaudeProviderWithOptions(WithClaudeNativeToolsDisabled()),
			expected: []string{"--output-format", "stream-json", "--verbose", "--system-prompt", "system", "--model", "claude-haiku-4-5-20251001", "--permission-mode", "bypassPermissions", "--tools", "", "--max-turns", "1", "-p", "hello"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			arguments := test.provider.streamCommandArgs("system", "hello", haikuModel())
			assertStringSlicesEqual(t, arguments, test.expected)
		})
	}
}

func TestNewClaudeProviderWithOptionsIgnoresNilAndPreservesOrder(t *testing.T) {
	var applied []string
	first := func(provider *ClaudeProvider) {
		applied = append(applied, "first")
		provider.permissionMode = "first"
	}
	second := func(provider *ClaudeProvider) {
		applied = append(applied, "second")
		provider.permissionMode = "second"
	}

	provider := NewClaudeProviderWithOptions(nil, first, nil, second, nil)

	assertStringSlicesEqual(t, applied, []string{"first", "second"})
	if provider.permissionMode != "second" {
		t.Fatalf("permission mode = %q, want second", provider.permissionMode)
	}
}

func assertStringSlicesEqual(t *testing.T, actual, expected []string) {
	t.Helper()
	if !slices.Equal(actual, expected) {
		t.Fatalf("arguments = %q, want %q", actual, expected)
	}
}

func haikuModel() sdk.ModelInfo {
	return claudeModels[2] // haiku
}
