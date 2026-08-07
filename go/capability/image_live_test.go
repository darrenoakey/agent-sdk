//go:build live_igs

package capability

import (
	"bytes"
	"context"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestImageServiceNativeTransportLive(t *testing.T) {
	requestContext, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	data, status, err := requestImageServiceBytes(requestContext, "GET", "/healthz", nil, "")
	if err != nil {
		t.Fatalf("native canonical image transport failed: %v", err)
	}
	if status != 200 || string(data) != "ok" {
		t.Fatalf("native canonical image health response status=%d body=%q", status, data)
	}
}

func TestGenerateImageLiveCanonical(t *testing.T) {
	const width = 256
	const height = 256
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(directory, "go-public-igs-canary.png")
	result, err := GenerateImage(context.Background(), "A cheerful blue robot waving on a clean white background", ImageOpts{
		Width: width, Height: height, Output: output,
		StatePath: filepath.Join(directory, "go-public-igs-canary.state.json"),
	})
	if err != nil {
		t.Fatalf("public canonical image call failed: %v", err)
	}
	if !result.Ready || result.Status != "done" || result.JobID == "" {
		t.Fatalf("public canonical image result is incomplete: %+v", result)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatalf("reading public canonical image artifact: %v", err)
	}
	if !bytes.HasPrefix(data, pngMagic) {
		t.Fatal("public canonical image artifact is not PNG")
	}
	config, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decoding public canonical image artifact: %v", err)
	}
	if config.Width != width || config.Height != height {
		t.Fatalf("public canonical image dimensions = %dx%d, want %dx%d", config.Width, config.Height, width, height)
	}
}
