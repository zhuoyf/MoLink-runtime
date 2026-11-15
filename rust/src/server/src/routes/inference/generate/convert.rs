// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_engine_core_client::protocol::multimodal::MmFeatures;
use vllm_text::{Prompt, TextDecodeOptions, TextRequest};

use super::types::GenerateRequest;
use super::validate;
use crate::error::ApiError;
use crate::lora::LoraModelResolution;
use crate::utils::{ResolvedRequestContext, merge_ec_transfer_params, merge_kv_transfer_params};

/// Lowered generate request plus the response request ID.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedRequest {
    pub request_id: String,
    pub text_request: TextRequest,
    pub stream: bool,
    /// Public response rendering options for route-layer helpers.
    pub options: ResponseOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct ResponseOptions {
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether the caller asked for usage on every streamed chunk.
    pub include_continuous_usage: bool,
    /// Whether the caller requested output logprobs on generate choices.
    pub include_logprobs: bool,
    /// Whether the caller requested top-level prompt logprobs.
    pub include_prompt_logprobs: bool,
}

/// Validate and lower one raw generate request into the internal
/// text-generation format.
pub(super) fn prepare_generate_request(
    request: GenerateRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    mm_features: Option<MmFeatures>,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;

    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);
    let include_logprobs = request.sampling_params.inner.logprobs.is_some();
    let include_prompt_logprobs = request.sampling_params.inner.prompt_logprobs.is_some();
    let mut sampling_params = request.sampling_params.inner;
    sampling_params.vllm_xargs = merge_kv_transfer_params(
        sampling_params.vllm_xargs,
        request.kv_transfer_params.as_ref(),
    );
    sampling_params.vllm_xargs = merge_ec_transfer_params(
        sampling_params.vllm_xargs,
        request.ec_transfer_params.as_ref(),
    );

    let text_request = TextRequest {
        request_id: ctx.request_id.clone(),
        prompt: Prompt::TokenIds(request.token_ids),
        mm_features,
        sampling_params,
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        prompt_truncation: None,
        priority: request.priority,
        cache_salt: request.cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id: ctx.session_id,
        reasoning_parser_kwargs: None,
        lora_request: lora_resolution.lora_request.clone(),
        arrival_time: None,
    };

    Ok(PreparedRequest {
        request_id: ctx.request_id,
        text_request,
        stream,
        options: ResponseOptions {
            include_usage,
            include_continuous_usage,
            include_logprobs,
            include_prompt_logprobs,
        },
    })
}
