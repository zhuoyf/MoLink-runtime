// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::ChatCompletionRequest;
use crate::error::{ApiError, bail_invalid_request};
use crate::routes::openai::utils::types::{ChatMessage, Tool};
use crate::routes::openai::utils::validate_generation_prompt_truncation;

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
    request: &ChatCompletionRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
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

    if request.top_logprobs.is_some() && !request.logprobs {
        bail_invalid_request!(
            param = "top_logprobs",
            "top_logprobs can only be used when logprobs=true."
        );
    }

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        if prompt_logprobs < 0 && prompt_logprobs != -1 {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs must be a non-negative value or -1."
            );
        }

        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs are not available when stream=true."
            );
        }
    }

    if let Some(tools) = request.tools.as_ref() {
        validate_function_tools(tools, "tools")?;
    }

    for message in &request.messages {
        if let ChatMessage::Developer {
            tools: Some(tools), ..
        } = message
        {
            validate_function_tools(tools, "messages[].tools")?;
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

    reject_non_default(
        request.length_penalty.as_ref(),
        "length_penalty",
        "length_penalty is not supported.",
    )?;
    if !request.spaces_between_special_tokens {
        bail_invalid_request!(
            param = "spaces_between_special_tokens",
            "spaces_between_special_tokens is not supported."
        );
    }
    validate_generation_prompt_truncation(request.truncate_prompt_tokens, request.echo)?;
    reject_non_default(
        request.media_io_kwargs.as_ref(),
        "media_io_kwargs",
        "media_io_kwargs is not supported.",
    )?;
    reject_non_default(
        request.mm_processor_kwargs.as_ref(),
        "mm_processor_kwargs",
        "mm_processor_kwargs is not supported.",
    )?;

    Ok(())
}

/// Reject one option unless it is entirely absent.
fn reject_non_default<T>(
    value: Option<&T>,
    param: &'static str,
    message: &str,
) -> Result<(), ApiError> {
    if value.is_some() {
        bail_invalid_request!(param = param, "{}", message);
    }
    Ok(())
}

fn validate_function_tools(tools: &[Tool], param: &'static str) -> Result<(), ApiError> {
    for tool in tools {
        if tool.tool_type != "function" {
            bail_invalid_request!(param = param, "Only function tools are supported.");
        }
    }
    Ok(())
}
