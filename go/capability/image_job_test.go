package capability

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func TestGetImageJobRejectsInvalidIdBeforeNetwork(t *testing.T) {
	for _, jobId := range []string{"", " ", "../alternate", "nested/job"} {
		status, err := GetImageJob(context.Background(), jobId)
		if status != nil || err == nil || !strings.Contains(err.Error(), "valid image job id") {
			t.Errorf("job id %q returned status=%+v error=%v", jobId, status, err)
		}
	}
}

func TestResumeImageJobRejectsTimeoutBeforeNetwork(t *testing.T) {
	result, err := ResumeImageJob(context.Background(), "durable-job", ImageJobOpts{Timeout: time.Second})
	var agentError *agentsdk.AgentError
	if result != nil || !errors.As(err, &agentError) || agentError.Kind != agentsdk.ErrorInvalidRequest {
		t.Fatalf("finite timeout result=%+v error=%v", result, err)
	}
	if !strings.Contains(err.Error(), "deadlines are actively disabled") {
		t.Fatalf("finite timeout guidance = %v", err)
	}
}

func TestTransientImageErrorsAreRecoverableWithoutDefaultDeadline(t *testing.T) {
	transient := agentsdk.NewAgentError("temporary status failure", agentsdk.ErrorNotAvailable, nil)
	terminal := agentsdk.NewAgentError("terminal status failure", agentsdk.ErrorInternal, nil)
	if !isTransientImageError(transient) || isTransientImageError(terminal) {
		t.Fatal("transient image error classification is incorrect")
	}
	if errors.Is(transient, context.DeadlineExceeded) {
		t.Fatal("default recovery introduced a finite deadline")
	}
}

func TestTerminalImageStatusPreservesIdentityAndNeverRecovers(t *testing.T) {
	terminal, err := classifyImageStatus("terminal-job", imageServiceStatus{Status: "failed", Error: "service failure"})
	if !terminal || err == nil {
		t.Fatalf("terminal status result: terminal=%v error=%v", terminal, err)
	}
	var agentError *agentsdk.AgentError
	if !errors.As(err, &agentError) || len(agentError.Attempts) != 1 {
		t.Fatalf("terminal status identity metadata missing: %v", err)
	}
	if agentError.Attempts[0]["job_id"] != "terminal-job" || agentError.Attempts[0]["recoverable"] != false {
		t.Fatalf("terminal status metadata = %+v", agentError.Attempts)
	}
}

func TestImageServiceResponseErrorTreatsMissingJobAsTerminal(t *testing.T) {
	err := imageServiceResponseError(404, []byte(`{"detail":"not found"}`))
	var agentError *agentsdk.AgentError
	if !errors.As(err, &agentError) || agentError.Kind != agentsdk.ErrorInternal {
		t.Fatalf("404 error = %v", err)
	}
	if isTransientImageError(err) {
		t.Fatalf("404 was recoverable: %v", err)
	}
	if len(agentError.Attempts) != 1 || agentError.Attempts[0]["recoverable"] != false {
		t.Fatalf("404 metadata = %+v", agentError.Attempts)
	}
}

func TestResumeImageJobShortCircuitsOnLocalArtifact(t *testing.T) {
	directory := t.TempDir()
	output := filepath.Join(directory, "done.png")
	writeTestPNG(t, output, 32, 24)

	// Parent is already canceled: any network attempt would fail immediately.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := ResumeImageJob(ctx, "local-only-job", ImageJobOpts{Output: output})
	if err != nil {
		t.Fatalf("local artifact short-circuit failed: %v", err)
	}
	if result == nil || result.Path != output || !result.Ready || result.Status != "done" {
		t.Fatalf("unexpected short-circuit result: %+v", result)
	}
	if result.Width != 32 || result.Height != 24 {
		t.Fatalf("dimensions = %dx%d", result.Width, result.Height)
	}
	if result.JobID != "local-only-job" {
		t.Fatalf("job id = %q", result.JobID)
	}
	if source, _ := result.Provenance["source"].(string); source != "local_artifact" {
		t.Fatalf("provenance source = %v", result.Provenance["source"])
	}
}

func TestDownloadImageJobPrefersLocalArtifactWithoutNetwork(t *testing.T) {
	directory := t.TempDir()
	output := filepath.Join(directory, "done.png")
	writeTestPNG(t, output, 16, 16)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := DownloadImageJob(ctx, "local-download-job", output, false)
	if err != nil {
		t.Fatalf("download local short-circuit failed: %v", err)
	}
	if result.Path != output || result.JobID != "local-download-job" {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestLoadCompletedLocalImageAcceptsJpegAndRejectsSymlink(t *testing.T) {
	directory := t.TempDir()
	jpegPath := filepath.Join(directory, "done.jpg")
	imageValue := image.NewRGBA(image.Rect(0, 0, 18, 12))
	imageValue.Set(0, 0, color.White)
	file, err := os.OpenFile(jpegPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatalf("create jpeg: %v", err)
	}
	if err := jpeg.Encode(file, imageValue, &jpeg.Options{Quality: 92}); err != nil {
		_ = file.Close()
		t.Fatalf("encode jpeg: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close jpeg: %v", err)
	}
	result, ok, err := loadCompletedLocalImage("jpeg-job", jpegPath)
	if err != nil || !ok || result.Width != 18 || result.Height != 12 {
		t.Fatalf("jpeg local artifact result=%+v ok=%v error=%v", result, ok, err)
	}
	wrongFormatPath := filepath.Join(directory, "wrong-format.png")
	if err := os.Link(jpegPath, wrongFormatPath); err != nil {
		t.Fatalf("link wrong-format artifact: %v", err)
	}
	if result, ok, err := loadCompletedLocalImage("wrong-format-job", wrongFormatPath); err != nil || ok || result != nil {
		t.Fatalf("wrong-format local artifact result=%+v ok=%v error=%v", result, ok, err)
	}

	linkedPath := filepath.Join(directory, "linked.jpg")
	if err := os.Symlink(jpegPath, linkedPath); err != nil {
		t.Fatalf("link jpeg: %v", err)
	}
	if result, ok, err := loadCompletedLocalImage("linked-job", linkedPath); err != nil || ok || result != nil {
		t.Fatalf("symlink local artifact result=%+v ok=%v error=%v", result, ok, err)
	}
}

func writeTestPNG(t *testing.T, path string, width, height int) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	if err := os.WriteFile(path, buffer.Bytes(), 0o600); err != nil {
		t.Fatalf("write png: %v", err)
	}
}
