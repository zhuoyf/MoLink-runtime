// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use uuid::Uuid;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::multimodal::MmFeatures;
use vllm_engine_core_client::protocol::request::{EngineCoreRequest, ReasoningParserKwargs};
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::error::{Error, Result};
use crate::request_metrics::current_unix_timestamp_secs;

/// Tokenized decoder-only generate request accepted by [`crate::Llm`].
///
/// This is the first-stage Rust subset of the inputs that eventually flow into
/// Python `AsyncLLM.generate()`. The boundary is intentionally above
/// [`EngineCoreRequest`], but below higher-level text and multimodal
/// preprocessing.
///
/// Original Python API reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/protocol.py#L67-L84>
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    /// Unique ID of the request.
    pub request_id: String,
    /// Token IDs of the prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Sampling parameters forwarded to engine-core.
    pub sampling_params: EngineCoreSamplingParams,
    /// Optional multimodal features already prepared by `vllm-chat`.
    pub mm_features: Option<MmFeatures>,
    /// Unix timestamp, in seconds, when this request arrived at the frontend.
    ///
    /// Stamped at the frontend entry, before render and tokenization, to match
    /// Python's renderer-entry arrival_time. When omitted, it is filled as a
    /// fallback before the request is sent to engine-core.
    pub arrival_time: Option<f64>,
    /// Optional salt used to partition prefix-cache entries for this request.
    pub cache_salt: Option<String>,
    /// Optional tracing headers to forward to engine-core and downstream
    /// observability hooks.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// Request scheduling priority. Lower values are scheduled earlier.
    pub priority: i32,
    /// Optional data-parallel rank override for routing this request.
    pub data_parallel_rank: Option<u32>,
    /// Stable session identity shared by related requests.
    pub session_id: Option<String>,
    /// Optional reasoning-parser kwargs forwarded to engine-side structured
    /// output logic.
    pub reasoning_parser_kwargs: Option<ReasoningParserKwargs>,
    /// Optional LoRA adapter request applied to this generation.
    pub lora_request: Option<LoraRequest>,
}

#[derive(Debug)]
pub(crate) struct PreparedGenerateRequest {
    pub engine_request: EngineCoreRequest,
}

impl GenerateRequest {
    /// Validate and lower this request into the raw engine-core request format.
    pub(crate) fn prepare(self, randomize_request_id: bool) -> Result<PreparedGenerateRequest> {
        if self.prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id,
            });
        }
        let GenerateRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            mm_features,
            arrival_time,
            cache_salt,
            trace_headers,
            priority,
            data_parallel_rank,
            session_id,
            reasoning_parser_kwargs,
            lora_request,
        } = self;

        let external_request_id = request_id;
        let engine_request_id = if randomize_request_id {
            let random_suffix = Uuid::new_v4().simple().to_string();
            format!("{external_request_id}-{}", &random_suffix[..8])
        } else {
            external_request_id.clone()
        };
        Ok(PreparedGenerateRequest {
            engine_request: EngineCoreRequest {
                request_id: engine_request_id,
                prompt_token_ids: Some(prompt_token_ids),
                mm_features,
                sampling_params: Some(sampling_params),
                pooling_params: None,
                arrival_time: arrival_time.unwrap_or_else(current_unix_timestamp_secs),
                lora_request,
                cache_salt,
                data_parallel_rank,
                prompt_embeds: None,
                prompt_is_token_ids: None,
                client_index: 0,
                current_wave: 0,
                priority,
                trace_headers,
                resumable: false,
                session_id,
                external_req_id: Some(external_request_id),
                // Rust parser doesn't expose this information, leave it unset and let the
                // reasoning logic in engine-sided structured output manager handle it.
                reasoning_ended: None,
                reasoning_parser_kwargs,
                abort_immediately: false,
            },
        })
    }
}

impl PreparedGenerateRequest {
    /// Return the original prompt token IDs copied into the raw engine request.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.engine_request
            .prompt_token_ids
            .as_ref()
            .expect("prepared request must have prompt token ids")
    }
}
