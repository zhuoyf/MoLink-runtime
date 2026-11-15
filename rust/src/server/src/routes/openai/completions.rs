// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod convert;
mod types;
mod validate;

use std::convert::Infallible;
use std::result::Result;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};
use tracing_futures::Instrument as _;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_text::tokenizer::Tokenizer;
use vllm_text::{
    DecodedPromptLogprobs, DecodedTextEvent, FinishReason, TextOutputStream,
    TextOutputStreamExt as _, TextRequest,
};

use self::convert::{ResponseOptions, prepare_completion_request};
pub(crate) use self::types::CompletionRequest;
use super::utils::logprobs::{
    collected_logprobs_to_openai, decoded_logprobs_to_openai, decoded_prompt_logprobs_to_openai,
    prompt_logprobs_to_maps, text_len,
};
use super::utils::types::{StreamResponseEnvelope, Usage};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, bail_server_error, server_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::completions::types::{
    CompletionChoice, CompletionResponse, CompletionSseChunk, CompletionStreamChoice,
    CompletionStreamResponse,
};
use crate::routes::openai::utils::types::LogProbs;
use crate::routes::openai::utils::usage::ContinuousUsage;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{ResolvedRequestContext, resolve_request_context, unix_timestamp};

pub(crate) fn lower_completion_request(
    request: CompletionRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    tokenizer: &dyn Tokenizer,
) -> Result<TextRequest, ApiError> {
    prepare_completion_request(request, lora_resolution, ctx, tokenizer)
        .map(|prepared| prepared.text_request)
}

/// Validate one completions request and proxy it into the shared `vllm-text`
/// stack.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;

    let tokenizer = state.chat.text().tokenizer();
    let prepared = match prepare_completion_request(
        body,
        &lora_resolution,
        request_context,
        tokenizer.as_ref(),
    ) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let api_server_options = state.api_server_options;
    let text_stream = match state
        .chat
        .text()
        .generate(prepared.text_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return text_submit_error("failed to submit completion request", error).into_response();
        }
    };

    if stream {
        let chunk_stream = completion_chunk_stream(
            text_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            api_server_options,
            prepared.options,
        );
        let sse_stream = completion_sse_stream(chunk_stream).instrument(request_span);

        Sse::new(sse_stream).into_response()
    } else {
        let response = match collect_completion(
            text_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            api_server_options,
            prepared.options,
        )
        .instrument(request_span.clone())
        .await
        {
            Ok(response) => response,
            Err(error) => return error.into_response(),
        };

        Json(response).into_response()
    }
}

async fn collect_completion(
    stream: impl TextOutputStream,
    request_id: String,
    response_model: String,
    created: u64,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        // Ignored: non-streaming responses always include usage.
        include_usage: _,
        // Ignored: non-streaming responses are collected before usage is attached.
        include_continuous_usage: _,
        prompt_only,
        echo,
        requested_logprobs,
        include_prompt_logprobs,
        return_token_ids,
        return_tokens_as_token_ids,
    }: ResponseOptions,
) -> Result<CompletionResponse, ApiError> {
    let collected = stream
        .collect_output()
        .await
        .map_err(|error| server_error!("completion stream failed: {}", error.to_report_string()))?;
    let finish_reason = collected.finish_reason.clone();
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);

    let prompt_char_count = echo.as_ref().map(|prompt| text_len(prompt)).unwrap_or_default();
    let logprobs = if requested_logprobs.is_some() && prompt_only {
        let prompt = echo.as_deref().ok_or_else(|| {
            server_error!("prompt-only completion response missing echoed prompt")
        })?;
        Some(prompt_only_logprobs_to_openai(
            collected.prompt_logprobs.as_ref(),
            prompt,
            collected.prompt_token_ids.as_ref(),
            return_tokens_as_token_ids,
        )?)
    } else if requested_logprobs.is_some() {
        Some(collected_logprobs_to_openai(
            &collected,
            echo.is_some(),
            prompt_char_count,
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(prompt_logprobs_to_maps(
            collected.prompt_logprobs.as_ref(),
            collected.prompt_token_ids.as_ref(),
        )?)
    } else {
        None
    };
    let text = match &echo {
        None => collected.text,
        Some(prompt) if prompt_only => prompt.clone(),
        Some(prompt) => format!("{prompt}{}", collected.text),
    };
    let finish_reason = completion_finish_reason_to_openai(&finish_reason)?.to_string();
    let usage = Usage::from_token_usage(collected.usage, enable_prompt_tokens_details);

    if enable_log_requests {
        info!(
            model = %response_model,
            prompt_tokens = usage.prompt_tokens,
            output_tokens = usage.completion_tokens.unwrap_or(0),
            %finish_reason,
            "completion finished"
        );
    }

    Ok(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model: response_model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs,
            finish_reason: Some(finish_reason),
            stop_reason,
            prompt_logprobs,
            token_ids: return_token_ids.then(|| collected.token_ids.clone()),
            prompt_token_ids: return_token_ids.then(|| collected.prompt_token_ids.to_vec()),
        }],
        usage: Some(usage),
        system_fingerprint: None,
        kv_transfer_params: collected.kv_transfer_params,
        ec_transfer_params: collected.ec_transfer_params,
    })
}

/// Convert one internal decoded-text stream into OpenAI completions chunks.
#[try_stream]
async fn completion_chunk_stream(
    stream: impl TextOutputStream,
    request_id: String,
    response_model: String,
    created: u64,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        include_usage,
        include_continuous_usage,
        prompt_only,
        echo,
        requested_logprobs,
        // Ignored: streaming prompt logprobs are rejected for Python parity.
        include_prompt_logprobs: _,
        return_token_ids,
        return_tokens_as_token_ids,
    }: ResponseOptions,
    mut y: TryYielder<CompletionSseChunk, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let envelope = Arc::new(StreamResponseEnvelope::new(
        request_id,
        "text_completion",
        created,
        response_model,
    ));
    let mut visible_text_len = 0_u32;
    let mut first_chunk = true;
    let mut continuous_usage = ContinuousUsage::default();

    /// Yield a chunk with optional continuous usage attached.
    macro_rules! yield_chunk {
        ($chunk:expr) => {{
            let mut chunk = $chunk;
            if include_continuous_usage {
                chunk.usage = Some(continuous_usage.to_usage());
            }
            y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
        }};
    }

    while let Some(next) = stream.next().await {
        match next {
            Ok(DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            }) => {
                debug!("completion stream started");
                continuous_usage.set_prompt_tokens(prompt_token_ids.len());
                if let Some(prompt) = echo.as_ref() {
                    visible_text_len = text_len(prompt);
                    let logprobs = if prompt_only && requested_logprobs.is_some() {
                        Some(prompt_only_logprobs_to_openai(
                            prompt_logprobs.as_ref(),
                            prompt,
                            prompt_token_ids.as_ref(),
                            return_tokens_as_token_ids,
                        )?)
                    } else {
                        None
                    };
                    let mut chunk = delta_chunk(&envelope, prompt.clone(), logprobs);
                    if return_token_ids && first_chunk {
                        if let Some(choice) = chunk.choices.first_mut() {
                            choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                        }
                        first_chunk = false;
                    }
                    yield_chunk!(chunk);
                } else if return_token_ids {
                    // Emit a chunk with prompt_token_ids in the first streaming response
                    let mut chunk = delta_chunk(&envelope, String::new(), None);
                    if let Some(choice) = chunk.choices.first_mut() {
                        choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                    }
                    first_chunk = false;
                    yield_chunk!(chunk);
                }
            }
            Ok(DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            }) => {
                // Prompt-only streaming already emitted the echoed prompt in the Start chunk.
                // The one generated token is only used to drive the engine to a finished event,
                // so hide its delta and forward only the terminal finish/usage metadata.
                if prompt_only {
                    if let Some(finished) = finished {
                        if enable_log_requests {
                            info!(
                                stream = true,
                                model = %envelope.model(),
                                prompt_tokens = finished.usage.prompt_token_count,
                                output_tokens = finished.usage.output_token_count,
                                finish_reason = finished.finish_reason.as_str(),
                                "completion finished"
                            );
                        }
                        continuous_usage.set_final_counts(
                            finished.usage.prompt_token_count,
                            finished.usage.output_token_count,
                        );
                        let final_chunk = final_chunk(&envelope, finished.finish_reason)?;
                        yield_chunk!(final_chunk);

                        if include_usage {
                            y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                                &envelope,
                                Usage::from_token_usage(
                                    finished.usage,
                                    enable_prompt_tokens_details,
                                ),
                            )))
                            .await;
                        }
                    }
                    continue;
                }
                let delta_text_len = text_len(&delta);
                let logprobs = if requested_logprobs.is_some() {
                    let decoded_logprobs = logprobs.as_ref().ok_or_else(|| {
                        server_error!(
                            "completion stream requested logprobs but generation returned none"
                        )
                    })?;
                    Some(decoded_logprobs_to_openai(
                        decoded_logprobs,
                        visible_text_len,
                        return_tokens_as_token_ids,
                    )?)
                } else {
                    None
                };
                let mut chunk = delta_chunk(&envelope, delta, logprobs);
                let delta_token_count = token_ids.len();
                continuous_usage.add_output_tokens(delta_token_count);
                if return_token_ids && let Some(choice) = chunk.choices.first_mut() {
                    choice.token_ids = Some(token_ids);
                }
                yield_chunk!(chunk);
                visible_text_len = visible_text_len.saturating_add(delta_text_len);

                if let Some(finished) = finished {
                    if enable_log_requests {
                        info!(
                            stream = true,
                            model = %envelope.model(),
                            prompt_tokens = finished.usage.prompt_token_count,
                            output_tokens = finished.usage.output_token_count,
                            finish_reason = finished.finish_reason.as_str(),
                            "completion finished"
                        );
                    }
                    continuous_usage.set_final_counts(
                        finished.usage.prompt_token_count,
                        finished.usage.output_token_count,
                    );
                    let final_chunk = final_chunk(&envelope, finished.finish_reason)?;
                    yield_chunk!(final_chunk);

                    if include_usage {
                        y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                            &envelope,
                            Usage::from_token_usage(finished.usage, enable_prompt_tokens_details),
                        )))
                        .await;
                    }
                }
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "completion stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
    Ok(())
}

fn delta_chunk(
    envelope: &Arc<StreamResponseEnvelope>,
    text: String,
    logprobs: Option<LogProbs>,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(envelope);
    chunk.choices.push(CompletionStreamChoice {
        text,
        logprobs,
        ..Default::default()
    });
    chunk
}

fn final_chunk(
    envelope: &Arc<StreamResponseEnvelope>,
    finish_reason: FinishReason,
) -> Result<CompletionStreamResponse, ApiError> {
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let finish_reason = completion_finish_reason_to_openai(&finish_reason)?;

    let mut chunk = CompletionStreamResponse::new(envelope);
    chunk.choices.push(CompletionStreamChoice {
        finish_reason: Some(finish_reason.to_string()),
        stop_reason,
        ..Default::default()
    });
    Ok(chunk)
}

fn completion_finish_reason_to_openai(
    finish_reason: &FinishReason,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop(_) => Ok("stop"),
        FinishReason::Repetition(_) => Ok("repetition"),
        FinishReason::Length => Ok("length"),
        FinishReason::Abort => Ok("abort"),
        FinishReason::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

fn stop_reason_to_json(stop_reason: &StopReason) -> Value {
    serde_json::to_value(stop_reason).expect("StopReason must serialize to JSON")
}

fn prompt_only_logprobs_to_openai(
    prompt_logprobs: Option<&DecodedPromptLogprobs>,
    prompt: &str,
    prompt_token_ids: &[u32],
    return_tokens_as_token_ids: bool,
) -> Result<LogProbs, ApiError> {
    if let Some(prompt_logprobs) = prompt_logprobs {
        return decoded_prompt_logprobs_to_openai(prompt_logprobs, 0, return_tokens_as_token_ids);
    }

    if let [token_id] = prompt_token_ids {
        let token = if return_tokens_as_token_ids {
            format!("token_id:{token_id}")
        } else {
            prompt.to_string()
        };

        return Ok(LogProbs {
            tokens: vec![token],
            token_logprobs: vec![None],
            top_logprobs: vec![None],
            text_offset: vec![0],
        });
    }

    Err(server_error!(
        "prompt-only completion requested logprobs but generation returned none"
    ))
}

fn usage_chunk(envelope: &Arc<StreamResponseEnvelope>, usage: Usage) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(envelope);
    chunk.usage = Some(usage);
    chunk
}

/// Convert one chunk stream into OpenAI-style SSE events.
///
/// OpenAI-style streaming errors are encoded as ordinary `data: {"error": ...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
async fn completion_sse_stream(
    stream: impl Stream<Item = Result<CompletionSseChunk, ApiError>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        match next {
            Ok(chunk) => y.yield_ok(to_sse_event(&chunk)).await,
            Err(error) => {
                y.yield_ok(to_error_sse_event(&error)).await;
                break;
            }
        }
    }

    y.yield_ok(done_sse_event()).await;
    Ok(())
}

/// Serialize one OpenAI chunk payload into one SSE `data:` event.
fn to_sse_event(chunk: &CompletionSseChunk) -> Event {
    trace!(?chunk, "completion emitting chunk");
    Event::default()
        .json_data(chunk)
        .expect("completion chunk must serialize to JSON")
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let response = error.to_error_response();
    trace!(?response, "completion emitting error");
    Event::default()
        .json_data(response)
        .expect("ErrorResponse must serialize to JSON")
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("completion emitting done");
    Event::default().data("[DONE]")
}
