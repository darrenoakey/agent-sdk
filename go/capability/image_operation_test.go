package capability

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestImageOperationProcessHelper(t *testing.T) {
	separator := -1
	for index, argument := range os.Args {
		if argument == "--" {
			separator = index
			break
		}
	}
	if separator < 0 || len(os.Args) != separator+5 {
		return
	}
	body := []byte(os.Args[separator+1])
	options := ImageOpts{Output: os.Args[separator+2], StatePath: os.Args[separator+3]}
	_, state, err := prepareImageOperation(body, options)
	if err != nil {
		t.Fatal(err)
	}
	if os.Args[separator+4] == "accept" {
		state.JobId = "accepted-job-id"
		if err := writeImageOperation(options.StatePath, state); err != nil {
			t.Fatal(err)
		}
	}
	data, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("IMAGE_OPERATION=%s\n", data)
}

func TestDefaultApiOperationSurvivesProcessBoundaryBeforeResponse(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	statePath := filepath.Join(directory, "operation.json")
	outputPath := filepath.Join(directory, "result.png")
	body := `{"prompt":"restart proof","width":64,"height":64,"transparent":false}`
	first := runImageOperationProcess(t, body, outputPath, statePath, "prepare")
	second := runImageOperationProcess(t, body, outputPath, statePath, "prepare")
	if first != second || first.JobId != "" || first.RequestBody != body || first.IdempotencyKey == "" {
		t.Fatalf("restart changed durable operation: first=%+v second=%+v", first, second)
	}
	accepted := runImageOperationProcess(t, body, outputPath, statePath, "accept")
	recovered := runImageOperationProcess(t, body, outputPath, statePath, "prepare")
	if accepted.JobId != "accepted-job-id" || recovered != accepted {
		t.Fatalf("accepted identity was not durable: accepted=%+v recovered=%+v", accepted, recovered)
	}
}

func TestExplicitKeyCreatesDeliberateRegenerationIdentity(t *testing.T) {
	body := []byte(`{"prompt":"same","width":64,"height":64,"transparent":false}`)
	intent := "path:/tmp/result.png"
	first := imageOperationId(body, intent, "operation-one")
	second := imageOperationId(body, intent, "operation-two")
	if first == second {
		t.Fatal("explicit operation keys did not create distinct regeneration identities")
	}
}

func TestImageOperationRejectsSymlinkAndUntrustedExistingState(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(directory, "target.json")
	statePath := filepath.Join(directory, "state.json")
	if err := os.WriteFile(target, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, statePath); err != nil {
		t.Fatal(err)
	}
	if _, err := readImageOperation(statePath); err == nil {
		t.Fatal("state symlink was accepted")
	}
	if err := os.Remove(statePath); err != nil {
		t.Fatal(err)
	}
	valid := imageOperationState{
		Version: imageOperationStateVersion, OperationId: "invalid", IdempotencyKey: "key",
		RequestBody: `{"prompt":"x"}`, OutputIntent: "path:" + filepath.Join(directory, "out.png"),
		OutputPath: filepath.Join(directory, "out.png"), OutputFormat: "png",
	}
	data, err := json.Marshal(valid)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(statePath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readImageOperation(statePath); err == nil {
		t.Fatal("non-owner-only state was accepted")
	}
	if err := os.Chmod(statePath, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readImageOperation(statePath); err == nil || !strings.Contains(err.Error(), "identity") {
		t.Fatalf("invalid immutable identity error = %v", err)
	}
}

func TestStateParentSymlinkIsRejectedBeforeCreatingBeneathIt(t *testing.T) {
	directory := t.TempDir()
	owned := filepath.Join(directory, "owned")
	if err := os.Mkdir(owned, 0o700); err != nil {
		t.Fatal(err)
	}
	linked := filepath.Join(directory, "linked")
	if err := os.Symlink(owned, linked); err != nil {
		t.Fatal(err)
	}
	statePath := filepath.Join(linked, "missing", "operation.json")
	body := []byte(`{"prompt":"symlink proof","width":64,"height":64}`)
	_, _, err := prepareImageOperation(body, ImageOpts{
		Output: filepath.Join(directory, "out.png"), StatePath: statePath,
	})
	if err == nil {
		t.Fatal("state parent symlink was accepted")
	}
	if _, statError := os.Lstat(filepath.Join(owned, "missing")); !os.IsNotExist(statError) {
		t.Fatalf("directory was created beneath rejected symlink: %v", statError)
	}
}

func runImageOperationProcess(t *testing.T, body, outputPath, statePath, action string) imageOperationState {
	t.Helper()
	command := exec.Command(os.Args[0], "-test.run=^TestImageOperationProcessHelper$", "--", body, outputPath, statePath, action)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("running image operation process: %v: %s", err, output)
	}
	for _, line := range strings.Split(string(output), "\n") {
		if !strings.HasPrefix(line, "IMAGE_OPERATION=") {
			continue
		}
		var state imageOperationState
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "IMAGE_OPERATION=")), &state); err != nil {
			t.Fatalf("decoding image operation process result: %v", err)
		}
		return state
	}
	t.Fatalf("image operation process returned no state: %s", output)
	return imageOperationState{}
}

func TestResumeImageOperationShortCircuitsOnLocalArtifact(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(directory, "artifact.png")
	writeOperationTestPNG(t, output, 20, 10)
	statePath := filepath.Join(directory, "state.json")
	body := []byte(`{"prompt":"x","width":20,"height":10}`)
	intent := "path:" + output
	key := "short-circuit-key"
	state := imageOperationState{
		Version:        imageOperationStateVersion,
		OperationId:    imageOperationId(body, intent, key),
		IdempotencyKey: key,
		JobId:          "short-circuit-job",
		RequestBody:    string(body),
		OutputIntent:   intent,
		OutputPath:     output,
		OutputFormat:   "png",
	}
	if err := writeImageOperation(statePath, state); err != nil {
		t.Fatalf("write state: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := ResumeImageOperation(ctx, statePath)
	if err != nil {
		t.Fatalf("operation short-circuit failed: %v", err)
	}
	if result.Path != output || result.IdempotencyKey != key || result.JobID != "short-circuit-job" {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func writeOperationTestPNG(t *testing.T, path string, width, height int) {
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
