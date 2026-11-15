// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::CompletionRequest;
use crate::error::{ApiError, bail_invalid_request};
use crate::routes::openai::utils::validate_generation_prompt_truncation;

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
    request: &CompletionRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    // This path is intentionally scoped to the minimum surface needed by
    // `vllm-bench` random workload compatibility, so unsupported legacy
    // completions features fail early here.
    if let Some(model) = request.model.as_ref().filter(|model| !model.is_empty())
        && !served_model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }

    if request.stream_options.is_some() && !request.stream {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.n.unwrap_or(1) > 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if request.max_tokens == Some(0) && !request.echo {
        bail_invalid_request!(
            param = "max_tokens",
            "max_tokens=0 is only supported when echo=true."
        );
    }

    if request.suffix.is_some() {
        bail_invalid_request!(param = "suffix", "suffix is not supported.");
    }

    if let Some(logprobs) = request.logprobs
        && logprobs < 0
        && logprobs != -1
    {
        bail_invalid_request!(
            param = "logprobs",
            "`logprobs` must be a non-negative value or -1."
        );
    }

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "`prompt_logprobs` are not available when `stream=true`."
            );
        }

        if prompt_logprobs < 0 && prompt_logprobs != -1 {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "`prompt_logprobs` must be a non-negative value or -1."
            );
        }
    }

    if request.use_beam_search {
        bail_invalid_request!(
            param = "use_beam_search",
            "use_beam_search is not supported."
        );
    }

    // ---- Reject parameters that are accepted for deserialization but not yet
    // implemented ----

    if request.length_penalty.is_some() {
        bail_invalid_request!(param = "length_penalty", "length_penalty is not supported.");
    }
    if !request.spaces_between_special_tokens {
        bail_invalid_request!(
            param = "spaces_between_special_tokens",
            "spaces_between_special_tokens is not supported."
        );
    }
    validate_generation_prompt_truncation(request.truncate_prompt_tokens, request.echo)?;

    Ok(())
}
