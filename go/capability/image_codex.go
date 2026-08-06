package capability

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
	"github.com/google/uuid"
	"golang.org/x/sys/unix"
)

const (
	imagePollInterval   = 2 * time.Second
	imageRequestTimeout = 60 * time.Second
	imageSubmitAttempts = 3
	maxImageBytes       = 128 << 20
)

var pngMagic = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

var codexModelInfo = agentsdk.ModelInfo{
	Provider: "codex", ModelID: "macmini-image-service", DisplayName: "Mac mini Codex image service",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityImage}, Tier: agentsdk.TierHigh,
}

type imageServiceStatus struct {
	Id             string           `json:"id"`
	Status         string           `json:"status"`
	Attempts       int              `json:"attempts"`
	Error          string           `json:"error"`
	Provider       string           `json:"provider"`
	PromptVersion  int              `json:"prompt_version"`
	AttemptHistory []map[string]any `json:"attempt_history"`
	CreatedAt      string           `json:"created_at"`
	UpdatedAt      string           `json:"updated_at"`
}

func encodeSourceImages(paths []string) ([]string, error) {
	encoded := make([]string, 0, len(paths))
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, agentsdk.NewAgentError("input image not found: "+path, agentsdk.ErrorInvalidRequest, nil)
		}
		if len(data) == 0 {
			return nil, agentsdk.NewAgentError("input image is empty: "+path, agentsdk.ErrorInvalidRequest, nil)
		}
		encoded = append(encoded, base64.StdEncoding.EncodeToString(data))
	}
	return encoded, nil
}

// SubmitImageJob creates a durable image job and requires a caller-owned recovery key.
func SubmitImageJob(parent context.Context, prompt string, options ImageOpts) (*agentsdk.ImageSubmission, error) {
	if err := validateImageRoute(options); err != nil {
		return nil, err
	}
	if strings.TrimSpace(prompt) == "" || options.Width <= 0 || options.Height <= 0 {
		return nil, agentsdk.NewAgentError("image prompt and positive width/height are required", agentsdk.ErrorInvalidRequest, nil)
	}
	if strings.TrimSpace(options.IdempotencyKey) == "" {
		return nil, agentsdk.NewAgentError("idempotency key is required for direct image submission", agentsdk.ErrorInvalidRequest, nil)
	}
	sources, err := encodeSourceImages(options.inputImages())
	if err != nil {
		return nil, err
	}
	return submitEncodedImageJob(parent, prompt, options, sources)
}

// RecoverImageSubmission replays the exact persisted request bytes with their original key.
func RecoverImageSubmission(parent context.Context, requestBody []byte, idempotencyKey string, configurations ...*agentsdk.Config) (*agentsdk.ImageSubmission, error) {
	if err := validateOptionalImageConfig(configurations); err != nil {
		return nil, err
	}
	if len(requestBody) == 0 || strings.TrimSpace(idempotencyKey) == "" {
		return nil, agentsdk.NewAgentError("persisted request body and idempotency key are required", agentsdk.ErrorInvalidRequest, nil)
	}
	var lastError error
	for range imageSubmitAttempts {
		submission, terminal, err := postImageServiceJob(parent, requestBody, idempotencyKey)
		if err == nil || terminal {
			return submission, err
		}
		lastError = err
	}
	attempts := []map[string]any{{"idempotency_key": idempotencyKey, "recoverable": true}}
	return nil, agentsdk.NewAgentError("image service submission remains recoverable: "+lastError.Error(), agentsdk.ErrorNotAvailable, attempts)
}

// WaitImageSubmission replays exact persisted bytes until accepted or permanently rejected.
func WaitImageSubmission(parent context.Context, requestBody []byte, idempotencyKey string, configurations ...*agentsdk.Config) (*agentsdk.ImageSubmission, error) {
	if err := validateOptionalImageConfig(configurations); err != nil {
		return nil, err
	}
	operationContext := durableImageContext(parent)
	for {
		submission, err := RecoverImageSubmission(operationContext, requestBody, idempotencyKey, configurations...)
		if err == nil {
			return submission, nil
		}
		var agentError *agentsdk.AgentError
		if !errors.As(err, &agentError) || agentError.Kind != agentsdk.ErrorNotAvailable {
			return nil, err
		}
		<-time.After(imagePollInterval)
	}
}

func encodeImageJobBody(prompt string, options ImageOpts, sources []string) ([]byte, error) {
	payload := map[string]any{
		"prompt": prompt, "width": options.Width, "height": options.Height, "transparent": options.Transparent,
	}
	if len(sources) > 0 {
		payload["source_images"] = sources
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encoding image service request: %w", err)
	}
	return body, nil
}

func submitEncodedImageJob(parent context.Context, prompt string, options ImageOpts, sources []string) (*agentsdk.ImageSubmission, error) {
	body, err := encodeImageJobBody(prompt, options, sources)
	if err != nil {
		return nil, fmt.Errorf("encoding image service request: %w", err)
	}
	return RecoverImageSubmission(parent, body, options.IdempotencyKey, options.Config)
}

func postImageServiceJob(parent context.Context, body []byte, key string) (*agentsdk.ImageSubmission, bool, error) {
	data, status, err := requestImageServiceBytes(parent, http.MethodPost, "/jobs", body, key)
	if err != nil {
		return nil, false, err
	}
	return parseImageSubmission(data, status, key)
}

func parseImageSubmission(data []byte, status int, key string) (*agentsdk.ImageSubmission, bool, error) {
	var response struct {
		Id             string `json:"id"`
		IdempotencyKey string `json:"idempotency_key"`
		Replayed       bool   `json:"replayed"`
		Error          string `json:"error"`
		Code           string `json:"code"`
	}
	if status == http.StatusConflict || status == http.StatusGone {
		return nil, true, terminalSubmissionError(status, data, key)
	}
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, true, agentsdk.NewAgentError("image service returned invalid submission JSON", agentsdk.ErrorInternal, nil)
	}
	if status != http.StatusAccepted || response.Id == "" || response.IdempotencyKey != key {
		attempts := []map[string]any{{"idempotency_key": key, "status": status, "recoverable": true}}
		return nil, true, agentsdk.NewAgentError("image service returned invalid durable submission identity", agentsdk.ErrorInternal, attempts)
	}
	return &agentsdk.ImageSubmission{JobID: response.Id, IdempotencyKey: key, Replayed: response.Replayed}, true, nil
}

func terminalSubmissionError(status int, data []byte, key string) error {
	var response struct {
		Id    string `json:"id"`
		Error string `json:"error"`
		Code  string `json:"code"`
	}
	_ = json.Unmarshal(data, &response)
	label := "conflict"
	if status == http.StatusGone {
		label = "expired"
	}
	if response.Error == "" {
		response.Error = fmt.Sprintf("image submission idempotency key %s (HTTP %d)", label, status)
	}
	if response.Code == "" {
		response.Code = "idempotency_" + label
	}
	attempts := []map[string]any{{"idempotency_key": key, "job_id": response.Id, "status": status, "code": response.Code, "recoverable": false}}
	return agentsdk.NewAgentError(response.Error, agentsdk.ErrorInvalidRequest, attempts)
}

func waitImageServiceJob(parent context.Context, jobId string) (imageServiceStatus, error) {
	ticker := time.NewTicker(imagePollInterval)
	defer ticker.Stop()
	for {
		status, err := fetchImageServiceStatus(parent, jobId)
		if err != nil {
			if parent.Err() != nil {
				return status, parent.Err()
			}
			if !isTransientImageError(err) {
				return status, err
			}
			<-ticker.C
			continue
		}
		terminal, terminalError := classifyImageStatus(jobId, status)
		if terminal {
			return status, terminalError
		}
		select {
		case <-parent.Done():
			return status, parent.Err()
		case <-ticker.C:
		}
	}
}

func waitImageServiceImage(parent context.Context, jobId string) ([]byte, error) {
	ticker := time.NewTicker(imagePollInterval)
	defer ticker.Stop()
	for {
		data, err := fetchImageServiceImage(parent, jobId)
		if err == nil {
			return data, nil
		}
		select {
		case <-parent.Done():
			return nil, parent.Err()
		case <-ticker.C:
		}
	}
}

func classifyImageStatus(jobId string, status imageServiceStatus) (bool, error) {
	switch status.Status {
	case "done":
		return true, nil
	case "failed":
		message := fmt.Sprintf("image service job %s failed after %d attempts: %s", jobId, status.Attempts, status.Error)
		attempts := []map[string]any{{"job_id": jobId, "status": status.Status, "recoverable": false}}
		return true, agentsdk.NewAgentError(message, agentsdk.ErrorInternal, attempts)
	case "cancelled", "canceled":
		attempts := []map[string]any{{"job_id": jobId, "status": status.Status, "recoverable": false}}
		return true, agentsdk.NewAgentError("image service job "+jobId+" was explicitly cancelled", agentsdk.ErrorInternal, attempts)
	case "queued", "running":
		return false, nil
	default:
		return true, agentsdk.NewAgentError(
			fmt.Sprintf("image service job %s returned unknown status %q", jobId, status.Status), agentsdk.ErrorInternal, nil,
		)
	}
}

func fetchImageServiceStatus(parent context.Context, jobId string) (imageServiceStatus, error) {
	var status imageServiceStatus
	err := imageServiceJson(parent, http.MethodGet, "/jobs/"+jobId, nil, &status)
	return status, err
}

func recoverableJobError(jobId, status string, cause error) error {
	attempts := []map[string]any{{"job_id": jobId, "status": status, "recoverable": true}}
	return agentsdk.NewAgentError("image service job "+jobId+" remains durable: "+cause.Error(), agentsdk.ErrorNotAvailable, attempts)
}

func imageServiceJson(parent context.Context, method, path string, payload any, target any) error {
	data, status, err := requestImageService(parent, method, path, payload)
	if err != nil {
		return err
	}
	if responseError := imageServiceResponseError(status, data); responseError != nil {
		return responseError
	}
	if err := json.Unmarshal(data, target); err != nil {
		return agentsdk.NewAgentError("image service returned invalid JSON: "+err.Error(), agentsdk.ErrorInternal, nil)
	}
	return nil
}

func imageServiceResponseError(status int, data []byte) error {
	if status >= 200 && status < 300 {
		return nil
	}
	recoverable := status == http.StatusRequestTimeout || status == http.StatusTooEarly || status == http.StatusTooManyRequests || status >= 500
	kind := agentsdk.ErrorInternal
	if recoverable {
		kind = agentsdk.ErrorNotAvailable
	}
	attempts := []map[string]any{{"http_status": status, "recoverable": recoverable}}
	return agentsdk.NewAgentError(fmt.Sprintf("image service returned HTTP %d: %s", status, data), kind, attempts)
}

func fetchImageServiceImage(parent context.Context, jobId string) ([]byte, error) {
	data, status, err := requestImageService(parent, http.MethodGet, "/jobs/"+jobId+"/image", nil)
	if err != nil {
		return nil, recoverableJobError(jobId, "done", err)
	}
	if status < 200 || status >= 300 {
		// Missing artifact (404) is terminal once the job record itself is gone or
		// no longer serves bytes. Callers must short-circuit on a local PNG first.
		if status == http.StatusNotFound || status == http.StatusGone {
			attempts := []map[string]any{{"job_id": jobId, "status": "done", "recoverable": false, "http_status": status}}
			return nil, agentsdk.NewAgentError(
				fmt.Sprintf("image service artifact for job %s is no longer available (HTTP %d)", jobId, status),
				agentsdk.ErrorInternal,
				attempts,
			)
		}
		return nil, recoverableJobError(jobId, "done", fmt.Errorf("artifact returned HTTP %d", status))
	}
	if err := validatePng(data); err != nil {
		return nil, recoverableJobError(jobId, "done", err)
	}
	return data, nil
}

func requestImageService(parent context.Context, method, path string, payload any) ([]byte, int, error) {
	var input []byte
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return nil, 0, fmt.Errorf("encoding image service request: %w", err)
		}
		input = encoded
	}
	return requestImageServiceBytes(parent, method, path, input, "")
}

// imageServiceHTTPClient talks to the durable Mac mini image route directly
// over Go's net/http instead of shelling out to curl. Forking a subprocess
// (exec.Command) from this CGO-linked binary is unsafe on Darwin: crypto/x509
// and the system DNS resolver route through CoreFoundation/Security.framework,
// which spin up background threads that are not fork-safe. If fork() lands
// while one of those threads holds an internal CF lock, the forked child can
// hang forever before it ever reaches execve() - silently, with no error and
// no syscall trace, exactly the failure mode that stalled image generation for
// hours. A native HTTP client never calls fork(), which removes the hazard
// entirely rather than trying to win the race.
var imageServiceHTTPClient = &http.Client{
	Timeout: imageRequestTimeout,
	Transport: &http.Transport{
		Proxy: nil, // never use a system/env proxy for the durable image route
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		// IdleConnTimeout keeps the pool from retaining dead LAN routes forever.
		IdleConnTimeout: 90 * time.Second,
	},
	CheckRedirect: func(request *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse // curl ran with --max-redirs 0
	},
}

func requestImageServiceBytes(parent context.Context, method, path string, input []byte, idempotencyKey string) ([]byte, int, error) {
	requestContext, cancel := context.WithTimeout(parent, imageRequestTimeout)
	defer cancel()
	var body io.Reader
	if input != nil {
		body = bytes.NewReader(input)
	}
	request, err := http.NewRequestWithContext(requestContext, method, "http://10.0.0.46:8830"+path, body)
	if err != nil {
		return nil, 0, agentsdk.NewAgentError("image service request could not be constructed: "+err.Error(), agentsdk.ErrorInternal, nil)
	}
	if input != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if idempotencyKey != "" {
		request.Header.Set("Idempotency-Key", idempotencyKey)
	}
	response, err := imageServiceHTTPClient.Do(request)
	if err != nil {
		return nil, 0, agentsdk.NewAgentError("image service transport failed: "+err.Error(), agentsdk.ErrorNotAvailable, nil)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, maxImageBytes+4096))
	if err != nil {
		return nil, 0, agentsdk.NewAgentError("image service transport failed: "+err.Error(), agentsdk.ErrorNotAvailable, nil)
	}
	return data, response.StatusCode, nil
}

func validatePng(data []byte) error {
	if len(data) == 0 || len(data) > maxImageBytes || !bytes.HasPrefix(data, pngMagic) {
		return agentsdk.NewAgentError("image service returned an empty or non-PNG artifact", agentsdk.ErrorInternal, nil)
	}
	decoded, format, err := image.Decode(bytes.NewReader(data))
	if err != nil || format != "png" || decoded.Bounds().Dx() <= 0 || decoded.Bounds().Dy() <= 0 {
		return agentsdk.NewAgentError("image service returned an invalid PNG artifact", agentsdk.ErrorInternal, nil)
	}
	return nil
}

func writeServiceImage(data []byte, output string, transparent bool) error {
	if err := os.MkdirAll(filepath.Dir(output), 0o700); err != nil {
		return fmt.Errorf("creating image output directory: %w", err)
	}
	if err := rejectSymlinkComponents(filepath.Dir(output)); err != nil {
		return err
	}
	if details, err := os.Lstat(output); err == nil {
		statData, ok := details.Sys().(*syscall.Stat_t)
		if !details.Mode().IsRegular() || !ok || statData.Uid != uint32(os.Geteuid()) {
			return fmt.Errorf("image output target is unsafe: %s", output)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("inspecting image output target: %w", err)
	}
	directory, err := openImageOperationDirectory(filepath.Dir(output))
	if err != nil {
		return fmt.Errorf("opening image output directory: %w", err)
	}
	defer directory.Close()
	temporaryName := "." + filepath.Base(output) + "." + uuid.NewString()
	descriptor, err := unix.Openat(int(directory.Fd()), temporaryName, unix.O_RDWR|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW, 0o600)
	if err != nil {
		return fmt.Errorf("creating image output temporary file: %w", err)
	}
	temporary := os.NewFile(uintptr(descriptor), temporaryName)
	defer func() { _ = unix.Unlinkat(int(directory.Fd()), temporaryName, 0) }()
	extension := strings.ToLower(filepath.Ext(output))
	if extension == ".jpg" || extension == ".jpeg" {
		if transparent {
			_ = temporary.Close()
			return agentsdk.NewAgentError("transparent image output must be PNG, not JPEG", agentsdk.ErrorInvalidRequest, nil)
		}
		err = writeJpegTo(data, temporary)
	} else {
		_, err = temporary.Write(data)
	}
	if err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("syncing image output temporary file: %w", err)
	}
	if err := validateImageOpenFile(temporary, extension); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("closing image output temporary file: %w", err)
	}
	if err := unix.Renameat(int(directory.Fd()), temporaryName, int(directory.Fd()), filepath.Base(output)); err != nil {
		return fmt.Errorf("publishing image output: %w", err)
	}
	return directory.Sync()
}

func writeJpegTo(data []byte, output io.Writer) error {
	decoded, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("decoding image service PNG: %w", err)
	}
	background := image.NewRGBA(decoded.Bounds())
	draw.Draw(background, background.Bounds(), &image.Uniform{C: color.White}, image.Point{}, draw.Src)
	draw.Draw(background, background.Bounds(), decoded, decoded.Bounds().Min, draw.Over)
	return jpeg.Encode(output, background, &jpeg.Options{Quality: 92})
}

func validateImageOpenFile(file *os.File, extension string) error {
	details, err := file.Stat()
	if err != nil || !details.Mode().IsRegular() || details.Mode().Perm() != 0o600 || details.Size() <= 0 || details.Size() > maxImageBytes {
		return fmt.Errorf("generated image temporary file is unsafe")
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewinding generated image temporary file: %w", err)
	}
	decoded, format, err := image.Decode(file)
	expected := "png"
	if extension == ".jpg" || extension == ".jpeg" {
		expected = "jpeg"
	}
	if err != nil || format != expected || decoded.Bounds().Dx() <= 0 || decoded.Bounds().Dy() <= 0 {
		return fmt.Errorf("generated image temporary file is invalid")
	}
	return nil
}
