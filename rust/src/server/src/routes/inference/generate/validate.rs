// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::GenerateRequest;
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust token generate
/// route.
pub(crate) fn validate_request_compat(
    request: &GenerateRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    if let Some(model) = request.model.as_ref()
        && !served_model_names.iter().any(|n| n == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }

    if request.stream_options.is_some() && !request.stream {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.sampling_params.n.unwrap_or(1) != 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if request.token_ids.is_empty() {
        bail_invalid_request!(
            param = "token_ids",
            "token_ids must contain at least one token ID."
        );
    }

    if request.sampling_params.inner.max_tokens == Some(0) {
        bail_invalid_request!(
            param = "sampling_params",
            "max_tokens must be greater than 0."
        );
    }

    if let Some(prompt_logprobs) = request.sampling_params.inner.prompt_logprobs {
        if prompt_logprobs < 0 && prompt_logprobs != -1 {
            bail_invalid_request!(
                param = "sampling_params",
                "`prompt_logprobs` must be a non-negative value or -1."
            );
        }

        if request.stream {
            bail_invalid_request!(
                param = "sampling_params",
                "`prompt_logprobs` are not available when `stream=true`."
            );
        }
    }

    Ok(())
}
