package capability

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
	"github.com/google/uuid"
	"golang.org/x/sys/unix"
)

// ImageJobOpts controls recovery of an existing durable image job.
type ImageJobOpts struct {
	Output      string
	Transparent bool
	Timeout     time.Duration
	Config      *agentsdk.Config
}

// GetImageJob returns durable IGS state without creating or changing a job.
func GetImageJob(parent context.Context, jobId string, configurations ...*agentsdk.Config) (*agentsdk.ImageJobStatus, error) {
	if err := validateOptionalImageConfig(configurations); err != nil {
		return nil, err
	}
	if strings.TrimSpace(jobId) == "" || strings.Contains(jobId, "/") {
		return nil, agentsdk.NewAgentError("a valid image job id is required", agentsdk.ErrorInvalidRequest, nil)
	}
	status, err := fetchImageServiceStatus(parent, jobId)
	if err != nil {
		return nil, err
	}
	if status.Id != "" && status.Id != jobId {
		return nil, agentsdk.NewAgentError("image service returned a mismatched job id", agentsdk.ErrorInternal, nil)
	}
	if _, err := classifyImageStatus(jobId, status); err != nil && status.Status != "failed" && status.Status != "cancelled" && status.Status != "canceled" {
		return nil, err
	}
	return publicImageJobStatus(jobId, status), nil
}

func publicImageJobStatus(jobId string, status imageServiceStatus) *agentsdk.ImageJobStatus {
	provider := status.Provider
	if provider == "" {
		provider = "codex"
	}
	provenance := map[string]any{
		"id": jobId, "status": status.Status, "attempts": status.Attempts,
		"error": status.Error, "provider": provider, "prompt_version": status.PromptVersion,
		"attempt_history": status.AttemptHistory, "created_at": status.CreatedAt, "updated_at": status.UpdatedAt,
	}
	return &agentsdk.ImageJobStatus{
		JobID: jobId, Status: status.Status, Ready: status.Status == "done", ModelUsed: codexModelInfo,
		Provider: provider, Attempts: status.Attempts, Error: status.Error, PromptVersion: status.PromptVersion,
		AttemptHistory: status.AttemptHistory, CreatedAt: status.CreatedAt, UpdatedAt: status.UpdatedAt, Provenance: provenance,
	}
}

// DownloadImageJob downloads and validates one completed IGS artifact.
func DownloadImageJob(parent context.Context, jobId, output string, transparent bool, configurations ...*agentsdk.Config) (*agentsdk.ImageResult, error) {
	if err := validateOptionalImageConfig(configurations); err != nil {
		return nil, err
	}
	// Prefer an already-materialized local artifact. Once the PNG is on disk the
	// durable operation is complete even if the remote job record has expired.
	if result, ok, err := loadCompletedLocalImage(jobId, output); err != nil {
		return nil, err
	} else if ok {
		return result, nil
	}
	status, err := GetImageJob(parent, jobId)
	if err != nil {
		return nil, err
	}
	if status.Status != "done" {
		if status.Status == "failed" || status.Status == "cancelled" || status.Status == "canceled" {
			attempts := []map[string]any{{"job_id": jobId, "status": status.Status, "recoverable": false}}
			return nil, agentsdk.NewAgentError(fmt.Sprintf("image service job %s ended with status %s: %s", jobId, status.Status, status.Error), agentsdk.ErrorInternal, attempts)
		}
		return nil, agentsdk.NewAgentError(fmt.Sprintf("image service job %s is %s, not done", jobId, status.Status), agentsdk.ErrorInvalidRequest, nil)
	}
	resolved, err := resolveOutputPath(output)
	if err != nil {
		return nil, err
	}
	data, err := fetchImageServiceImage(parent, jobId)
	if err != nil {
		return nil, err
	}
	if err := writeServiceImage(data, resolved, transparent); err != nil {
		return nil, err
	}
	configuration, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("reading validated image dimensions: %w", err)
	}
	return &agentsdk.ImageResult{
		Path: resolved, ModelUsed: codexModelInfo, ConversationID: uuid.New(), Width: configuration.Width,
		Height: configuration.Height, JobID: jobId, Status: "done", Ready: true, Provider: status.Provider,
		Provenance: status.Provenance,
	}, nil
}

// WaitImageJob waits through transient service failures and downloads only a terminal artifact.
func WaitImageJob(parent context.Context, jobId string, options ImageJobOpts) (*agentsdk.ImageResult, error) {
	return ResumeImageJob(parent, jobId, options)
}

// ResumeImageJob waits for an existing IGS id and never submits a replacement.
func ResumeImageJob(parent context.Context, jobId string, options ImageJobOpts) (*agentsdk.ImageResult, error) {
	if err := options.Config.ValidateImageConfig(); err != nil {
		return nil, err
	}
	if err := agentsdk.ValidateImageTimeout(options.Timeout); err != nil {
		return nil, err
	}
	// Short-circuit before any network call when the durable local artifact is
	// already present. This unblocks recovery after the remote job record is
	// gone (HTTP 404) or the transport path is wedged.
	if result, ok, err := loadCompletedLocalImage(jobId, options.Output); err != nil {
		return nil, err
	} else if ok {
		return result, nil
	}
	operationContext := durableImageContext(parent)
	for {
		status, err := GetImageJob(operationContext, jobId)
		if err != nil {
			// Remote job may have been reaped while the local PNG survived.
			if result, ok, loadErr := loadCompletedLocalImage(jobId, options.Output); loadErr != nil {
				return nil, loadErr
			} else if ok {
				return result, nil
			}
			if !isTransientImageError(err) {
				return nil, err
			}
			<-time.After(imagePollInterval)
			continue
		}
		if status.Status == "done" {
			result, downloadError := DownloadImageJob(operationContext, jobId, options.Output, options.Transparent)
			if downloadError == nil {
				return result, nil
			}
			if result, ok, loadErr := loadCompletedLocalImage(jobId, options.Output); loadErr != nil {
				return nil, loadErr
			} else if ok {
				return result, nil
			}
			if !isTransientImageError(downloadError) {
				return nil, downloadError
			}
		}
		if status.Status == "failed" || status.Status == "cancelled" || status.Status == "canceled" {
			// Prefer a completed local artifact over a later remote failure/cancel.
			if result, ok, loadErr := loadCompletedLocalImage(jobId, options.Output); loadErr != nil {
				return nil, loadErr
			} else if ok {
				return result, nil
			}
			attempts := []map[string]any{{"job_id": jobId, "status": status.Status, "recoverable": false}}
			return nil, agentsdk.NewAgentError(fmt.Sprintf("image service job %s ended with status %s: %s", jobId, status.Status, status.Error), agentsdk.ErrorInternal, attempts)
		}
		<-time.After(imagePollInterval)
	}
}

// loadCompletedLocalImage returns a ready ImageResult when output already holds
// a validated artifact in the format selected by its extension. ok=false means
// no usable local artifact (missing, unsafe, empty, or invalid).
func loadCompletedLocalImage(jobId, output string) (*agentsdk.ImageResult, bool, error) {
	if strings.TrimSpace(output) == "" {
		return nil, false, nil
	}
	resolved, err := resolveOutputPath(output)
	if err != nil {
		// resolveOutputPath only fails on empty or temp-create problems; empty is handled above.
		return nil, false, err
	}
	directory, err := openImageOperationDirectory(filepath.Dir(resolved))
	if err != nil {
		if errors.Is(err, syscall.ENOENT) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("opening local image artifact directory: %w", err)
	}
	defer directory.Close()
	descriptor, err := unix.Openat(int(directory.Fd()), filepath.Base(resolved), unix.O_RDONLY|unix.O_NOFOLLOW, 0)
	if err != nil {
		if errors.Is(err, syscall.ENOENT) || errors.Is(err, syscall.ELOOP) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("opening local image artifact: %w", err)
	}
	file := os.NewFile(uintptr(descriptor), resolved)
	defer file.Close()
	details, err := file.Stat()
	if err != nil {
		return nil, false, fmt.Errorf("inspecting local image artifact: %w", err)
	}
	statData, ok := details.Sys().(*syscall.Stat_t)
	if !ok || statData.Uid != uint32(os.Geteuid()) {
		return nil, false, nil
	}
	extension := strings.ToLower(filepath.Ext(resolved))
	if err := validateImageOpenFile(file, extension); err != nil {
		return nil, false, nil
	}
	if _, err := file.Seek(0, 0); err != nil {
		return nil, false, fmt.Errorf("rewinding local image artifact: %w", err)
	}
	configuration, _, err := image.DecodeConfig(file)
	if err != nil {
		return nil, false, nil
	}
	provider := "codex"
	provenance := map[string]any{
		"id": jobId, "status": "done", "source": "local_artifact",
		"provider": provider, "path": resolved,
	}
	return &agentsdk.ImageResult{
		Path: resolved, ModelUsed: codexModelInfo, ConversationID: uuid.New(),
		Width: configuration.Width, Height: configuration.Height, JobID: jobId,
		Status: "done", Ready: true, Provider: provider, Provenance: provenance,
	}, true, nil
}

func validateOptionalImageConfig(configurations []*agentsdk.Config) error {
	for _, configuration := range configurations {
		if err := configuration.ValidateImageConfig(); err != nil {
			return err
		}
	}
	return nil
}

func isTransientImageError(err error) bool {
	var agentError *agentsdk.AgentError
	return errors.As(err, &agentError) && agentError.Kind == agentsdk.ErrorNotAvailable
}
