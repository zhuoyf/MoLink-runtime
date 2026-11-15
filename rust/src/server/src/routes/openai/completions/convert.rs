// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror_ext::AsReport as _;
use vllm_text::tokenizer::Tokenizer;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextRequest};

use super::types::CompletionRequest;
use crate::error::{ApiError, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::completions::validate;
use crate::routes::openai::utils::resolve_generation_prompt_truncation;
use crate::routes::openai::utils::structured_outputs::convert_from_response_format_value;
use crate::utils::{
    ResolvedRequestContext, convert_logit_bias, merge_ec_transfer_params, merge_kv_transfer_params,
    resolve_session_id,
};

/// Lowered completion request plus the public response metadata carried by
/// every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedRequest {
    /// Stable OpenAI-style request ID, reused as the external text request ID.
    pub request_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Public response rendering options for route-layer helpers.
    pub options: ResponseOptions,
    /// Lowered text request for the shared `vllm-text` facade.
    pub text_request: TextRequest,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct ResponseOptions {
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether every streamed chunk should carry cumulative usage.
    pub include_continuous_usage: bool,
    /// Whether the caller requested prompt-only echo via `max_tokens=0`.
    pub prompt_only: bool,
    /// Prompt text that should be echoed back northbound when `echo=true`.
    pub echo: Option<String>,
    /// Whether the caller requested output logprobs on completion choices.
    pub requested_logprobs: Option<i32>,
    /// Whether the caller requested choice-level prompt logprobs.
    pub include_prompt_logprobs: bool,
    /// Whether to include token IDs alongside generated text.
    pub return_token_ids: bool,
    /// Whether to format logprob tokens as `token_id:{id}`.
    pub return_tokens_as_token_ids: bool,
}

/// Validate and lower one OpenAI completions request into the internal
/// text-generation format.
///
/// `lora_resolution.model_names` must be non-empty; the first entry is used as
/// the base `model` field in responses when no LoRA adapter is selected.
pub(super) fn prepare_completion_request(
    request: CompletionRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    tokenizer: &dyn Tokenizer,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;

    let prompt_truncation = resolve_generation_prompt_truncation(
        request.truncate_prompt_tokens,
        request.truncation_side,
    )
    .map_err(|error| text_submit_error("invalid prompt truncation", error))?;

    let request_id = format!("cmpl-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());

    let logprobs = request.logprobs;
    let prompt_only = request.echo && request.max_tokens == Some(0);
    let prompt_logprobs =
        request.prompt_logprobs.or(if request.echo && (!request.stream || prompt_only) {
            logprobs
        } else {
            None
        });
    let include_usage = (request.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);
    let include_prompt_logprobs = prompt_logprobs.is_some();
    let max_tokens = if prompt_only {
        Some(1)
    } else {
        request.max_tokens
    };
    let echo = completion_echo_text(&request, tokenizer)?;

    let structured_outputs =
        convert_from_response_format_value(&request.response_format, &request.structured_outputs)?;
    let session_id = resolve_session_id(
        &ctx,
        request.session_id.as_deref(),
        request.vllm_xargs.as_ref(),
    );

    let text_request = TextRequest {
        request_id: request_id.clone(),
        prompt: request.prompt,
        mm_features: None,
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: request.seed,
            max_tokens,
            min_tokens: request.min_tokens,
            thinking_token_budget: request.thinking_token_budget,
            logprobs,
            prompt_logprobs,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            repetition_detection: request.repetition_detection,
            stop_token_ids: request.stop_token_ids,
            ignore_eos: request.ignore_eos,
            logit_bias: convert_logit_bias(request.logit_bias)?,
            allowed_token_ids: request.allowed_token_ids,
            bad_words: None,
            logprob_token_ids: None,
            structured_outputs,
            skip_reading_prefix_cache: None,
            vllm_xargs: merge_kv_transfer_params(
                merge_ec_transfer_params(request.vllm_xargs, request.ec_transfer_params.as_ref()),
                request.kv_transfer_params.as_ref(),
            ),
        },
        decode_options: TextDecodeOptions {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
            stop_strings: request.stop.map(|stop| stop.into_vec()),
            min_tokens: request.min_tokens.unwrap_or(0),
        },
        intermediate: request.stream,
        prompt_truncation,
        priority: ctx.priority.or(request.priority).unwrap_or(0),
        cache_salt: request.cache_salt,
        add_special_tokens: request.add_special_tokens,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id,
        reasoning_parser_kwargs: None,
        lora_request: lora_resolution.lora_request.clone(),
        arrival_time: None,
    };

    Ok(PreparedRequest {
        request_id,
        response_model,
        options: ResponseOptions {
            include_usage,
            include_continuous_usage,
            prompt_only,
            echo,
            requested_logprobs: request.logprobs,
            include_prompt_logprobs,
            return_token_ids: request.return_token_ids.unwrap_or(false),
            return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
        },
        text_request,
    })
}

fn completion_echo_text(
    request: &CompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<Option<String>, ApiError> {
    if !request.echo {
        return Ok(None);
    }

    match &request.prompt {
        Prompt::Text(prompt) => Ok(Some(prompt.clone())),
        Prompt::TokenIds(token_ids) if request.return_token_ids.unwrap_or(false) => {
            Ok(Some(String::new()))
        }
        Prompt::TokenIds(token_ids) => {
            tokenizer.decode(token_ids, false).map(Some).map_err(|error| {
                ApiError::invalid_request(
                    format!(
                        "Failed to decode token-ID prompt for echo: {}",
                        error.to_report_string()
                    ),
                    Some("prompt"),
                )
            })
        }
    }
}
