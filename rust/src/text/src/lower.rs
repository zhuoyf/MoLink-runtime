// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;

pub(crate) mod logprobs;
pub(crate) mod sampling;
pub(crate) mod token_ids;

use logprobs::validate_logprobs;
use sampling::validate_resolved_sampling_params;
use token_ids::{validate_prompt_token_ids, validate_vocab_range};
use vllm_engine_core_client::protocol::sampling::{
    EngineCoreSamplingParams, RepetitionDetectionParams,
};
use vllm_llm::GenerateRequest;
use vllm_tokenizer::Tokenizer;

use crate::backend::{SamplingHints, SamplingLimits};
use crate::error::{Error, Result};
use crate::request::{SamplingParams, TextRequest};

/// One text request after it has been lowered into the raw generate boundary.
#[derive(Debug)]
pub struct PreparedTextRequest {
    /// The high-level request fields still needed for response-side metadata
    /// and decoding options.
    pub text_request: TextRequest,
    /// The southbound request ready to be sent to `vllm-llm`.
    pub generate_request: GenerateRequest,
}

/// Convert a high-level [`TextRequest`] into one lower-level
/// [`GenerateRequest`] ready for the `llm` crate.
pub fn lower_text_request(
    mut request: TextRequest,
    prompt_token_ids: Vec<u32>,
    sampling_hints: SamplingHints,
    sampling_limits: SamplingLimits,
    tokenizer: &dyn Tokenizer,
) -> Result<PreparedTextRequest> {
    let prompt_len = prompt_token_ids.len() as u32;
    validate_prompt_token_ids(&prompt_token_ids, &sampling_limits)?;

    let generate_request = GenerateRequest {
        request_id: request.request_id.clone(),
        prompt_token_ids,
        // Align with Python's response path: decoded output state does not retain
        // `mm_features`; move them to the engine request to avoid cloning large
        // multimodal tensor payloads.
        mm_features: request.mm_features.take(),
        sampling_params: lower_sampling_params(
            request.sampling_params.clone(),
            sampling_hints,
            sampling_limits,
            prompt_len,
            tokenizer,
        )?,
        cache_salt: request.cache_salt.clone(),
        priority: request.priority,
        data_parallel_rank: request.data_parallel_rank,
        session_id: request.session_id.clone(),
        reasoning_parser_kwargs: request.reasoning_parser_kwargs.clone(),
        lora_request: request.lora_request.clone(),
        arrival_time: request.arrival_time,
        trace_headers: None,
    };

    Ok(PreparedTextRequest {
        text_request: request,
        generate_request,
    })
}

/// Convert [`SamplingParams`] into [`EngineCoreSamplingParams`], enriching
/// omitted user values with tokenizer/model-derived hints when available.
pub fn lower_sampling_params(
    sampling_params: SamplingParams,
    SamplingHints {
        primary_eos_token_id,
        extra_eos_token_ids,
        default_temperature,
        default_top_p,
        default_top_k,
        default_min_p,
        default_repetition_penalty,
        default_max_tokens,
    }: SamplingHints,
    sampling_limits: SamplingLimits,
    prompt_len: u32,
    tokenizer: &dyn Tokenizer,
) -> Result<EngineCoreSamplingParams> {
    let SamplingParams {
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        thinking_token_budget,
        logprobs,
        prompt_logprobs,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_detection,
        stop_token_ids,
        ignore_eos,
        logit_bias,
        allowed_token_ids,
        bad_words,
        logprob_token_ids,
        structured_outputs,
        skip_reading_prefix_cache,
        vllm_xargs,
    } = sampling_params;

    validate_logprobs(
        logprobs,
        prompt_logprobs,
        logprob_token_ids.as_deref(),
        sampling_limits,
    )?;
    validate_repetition_detection(repetition_detection.as_ref())?;

    // Mirrors the model-generation-config inheritance used by vLLM's OpenAI chat
    // path: https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/openai/chat_completion/protocol.py#L424-L450
    // If neither the caller nor the model provides a value, fall back to 1.0 — the
    // default used by the Python vLLM OpenAI-compatible API (via
    // `_DEFAULT_SAMPLING_PARAMS`).
    let temperature = temperature.or(default_temperature).unwrap_or(1.0);
    let top_p = top_p.or(default_top_p).unwrap_or(1.0);
    let top_k = top_k.or(default_top_k).unwrap_or(0);
    let min_p = min_p.or(default_min_p).unwrap_or(0.0);
    let repetition_penalty = repetition_penalty.or(default_repetition_penalty).unwrap_or(1.0);
    let max_tokens = resolve_max_tokens(
        max_tokens,
        default_max_tokens,
        sampling_limits.max_model_len,
        prompt_len,
    )?;
    let min_tokens = min_tokens.unwrap_or(0);
    if min_tokens > max_tokens {
        return Err(Error::MinTokensExceedsMaxTokens {
            min_tokens,
            max_tokens,
        });
    }
    let thinking_token_budget = normalize_thinking_token_budget(thinking_token_budget)?;
    let frequency_penalty = frequency_penalty.unwrap_or(0.0);
    let presence_penalty = presence_penalty.unwrap_or(0.0);

    let mut stop_token_ids = stop_token_ids.unwrap_or_default();
    let mut all_stop_token_ids = BTreeSet::from_iter(stop_token_ids.iter().copied());
    if let Some(primary_eos_token_id) = primary_eos_token_id {
        all_stop_token_ids.insert(primary_eos_token_id);
    }
    all_stop_token_ids.extend(extra_eos_token_ids.iter().copied());

    if !ignore_eos {
        merge_unique_token_ids(&mut stop_token_ids, extra_eos_token_ids.iter().copied());
    }

    let params = EngineCoreSamplingParams {
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        thinking_token_budget,
        logprobs,
        prompt_logprobs,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_detection: repetition_detection.filter(|p| !p.is_disabled()),
        stop_token_ids,
        eos_token_id: (!ignore_eos).then_some(primary_eos_token_id).flatten(),
        all_stop_token_ids,
        logit_bias,
        allowed_token_ids,
        bad_words_token_ids: tokenize_bad_words(bad_words.as_deref(), tokenizer)?,
        // TODO: Validate structured-output schemas and regexes before submitting requests to engine-core.
        structured_outputs,
        logprob_token_ids,
        skip_reading_prefix_cache,
        extra_args: vllm_xargs,
        routed_experts_prompt_start: 0,
    };
    validate_resolved_sampling_params(&params)?;
    validate_vocab_range(&params, &sampling_limits)?;
    Ok(params)
}

/// Normalize the user-facing `thinking_token_budget` into the engine value.
///
/// Mirrors Python's `validate_thinking_token_budget`
/// (<https://github.com/vllm-project/vllm/blob/ecf9d83520eb217401b47d8a5451a27c5231b8c2/vllm/sampling_params.py#L35-L55>):
/// `None` and the `-1` "unlimited" sentinel both map to `None`; any other
/// negative value is rejected; non-negative values pass through unchanged. Like
/// Python's `int`, no upper bound is imposed.
fn normalize_thinking_token_budget(value: Option<i64>) -> Result<Option<u64>> {
    match value {
        None | Some(-1) => Ok(None),
        Some(budget) if budget >= 0 => Ok(Some(budget as u64)),
        Some(_) => Err(Error::InvalidThinkingTokenBudget),
    }
}

fn validate_repetition_detection(params: Option<&RepetitionDetectionParams>) -> Result<()> {
    let Some(params) = params else {
        return Ok(());
    };

    if params.min_pattern_size > params.max_pattern_size {
        return Err(Error::InvalidRepetitionDetection {
            message: format!(
                "`min_pattern_size` must be less than or equal to \
                 `max_pattern_size`, got min_pattern_size={}, \
                 max_pattern_size={}",
                params.min_pattern_size, params.max_pattern_size
            ),
        });
    }
    if params.max_pattern_size > 0 && params.min_count < 2 {
        return Err(Error::InvalidRepetitionDetection {
            message: format!(
                "`min_count` must be at least 2, got min_count={}",
                params.min_count
            ),
        });
    }

    Ok(())
}

/// Convert bad-word strings into token-ID sequences, following the Python vLLM
/// logic in `SamplingParams.update_from_tokenizer()`.
///
/// Each word is encoded both with and without a leading space so that the ban
/// applies regardless of whether the word appears at the beginning or in the
/// middle of generated text (this accounts for tokenizers that use an
/// `add_prefix_space` convention).
///
/// Reference: <https://github.com/vllm-project/vllm/blob/f22d6e026/vllm/sampling_params.py#L555-L594>
fn tokenize_bad_words(
    bad_words: Option<&[String]>,
    tokenizer: &dyn Tokenizer,
) -> Result<Option<Vec<Vec<u32>>>> {
    let bad_words = bad_words.filter(|w| !w.is_empty());
    let mut all_token_ids = Vec::new();

    for bad_word in bad_words.into_iter().flatten() {
        // Without a leading space we always keep the encoding.
        // With a leading space we only keep it when the prefix-space variant produces a
        // distinct first token but the same sequence length — this mirrors the Python
        // dedup condition that avoids redundant entries.
        let without_space = tokenizer.encode(bad_word, false)?;
        let with_space = tokenizer.encode(&format!(" {}", bad_word.trim_start()), false)?;

        if !without_space.is_empty() {
            all_token_ids.push(without_space);
        }
        if !with_space.is_empty()
            && all_token_ids.last().is_some_and(|prev: &Vec<u32>| {
                with_space[0] != prev[0] && with_space.len() == prev.len()
            })
        {
            all_token_ids.push(with_space);
        }
    }

    Ok((!all_token_ids.is_empty()).then_some(all_token_ids))
}

/// Resolve the effective `max_tokens` for generation, mirroring vLLM Python's
/// `get_max_tokens()` in `vllm/entrypoints/utils.py`.
///
/// Takes the minimum of all available limits: user-specified, generation-config
/// default, and `max_model_len - prompt_len`.
pub fn resolve_max_tokens(
    user_max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    max_model_len: u32,
    prompt_len: u32,
) -> Result<u32> {
    let model_max_tokens = if prompt_len >= max_model_len {
        return Err(Error::PromptTooLong {
            max_model_len,
            prompt_len,
        });
    } else {
        max_model_len - prompt_len
    };

    let request_max_tokens = user_max_tokens.or(default_max_tokens);
    Ok(request_max_tokens.map_or(model_max_tokens, |n| n.min(model_max_tokens)))
}

fn merge_unique_token_ids(
    stop_token_ids: &mut Vec<u32>,
    extra_token_ids: impl Iterator<Item = u32>,
) {
    // Keep user-provided ordering stable while still folding in backend-derived EOS
    // aliases.
    for token_id in extra_token_ids {
        if !stop_token_ids.contains(&token_id) {
            stop_token_ids.push(token_id);
        }
    }
}
