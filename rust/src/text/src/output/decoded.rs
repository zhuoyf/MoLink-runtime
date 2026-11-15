// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, trace};
use vllm_engine_core_client::AbortCause;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::{FinishReason, GenerateOutput, TokenUsage};
use vllm_tokenizer::{DynTokenizer, IncrementalDecoder};

use super::logprobs::{
    DecodedLogprobs, DecodedPromptLogprobs, decode_logprobs, decode_prompt_logprobs,
};
use crate::error::Error;

/// Request-neutral options for incremental text decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecodeOptions {
    pub skip_special_tokens: bool,
    pub include_stop_str_in_output: bool,
    pub stop_strings: Option<Vec<String>>,
    /// Minimum number of tokens to generate before stop-string checking kicks
    /// in. Stop strings found within the first `min_tokens` tokens are
    /// ignored.
    pub min_tokens: u32,
}

impl Default for TextDecodeOptions {
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            stop_strings: None,
            min_tokens: 0,
        }
    }
}

/// Terminal metadata carried on the final [`DecodedTextEvent`].
#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
    pub usage: TokenUsage,
    pub finish_reason: FinishReason,
    /// Connector-specific KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<serde_json::Value>,
    /// Connector-specific encoder cache transfer parameters for disaggregated
    /// serving.
    pub ec_transfer_params: Option<serde_json::Value>,
}

/// Internal decoded-text event emitted before higher-level assistant
/// adaptation.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTextEvent {
    /// The request has reached the point where prompt-scoped decoding metadata
    /// is ready.
    Start {
        /// The actual prompt token IDs for this request.
        prompt_token_ids: Arc<[u32]>,
        /// Once-only prompt logprobs metadata, when requested.
        ///
        /// The first prompt token is carried separately because it has no left
        /// context to score against; `scored_positions` covers the
        /// remaining prompt positions.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    /// A delta of text has been decoded, optionally alongside token-position
    /// logprobs.
    ///
    /// `delta` is the newly visible decoded text fragment for this update.
    ///
    /// `logprobs` covers the newly generated token positions from the same
    /// update, but is not guaranteed to align with `delta` by character
    /// span. One update may carry token logprobs but no newly visible text
    /// yet, and one visible text fragment may reflect multiple token
    /// positions becoming decodable together.
    ///
    /// Upper-level may further parse `delta` as reasoning or tool calls.
    ///
    /// When `finished` is `Some`, this is the terminal event for the request.
    TextDelta {
        delta: String,
        token_ids: Vec<u32>,
        logprobs: Option<DecodedLogprobs>,
        finished: Option<Finished>,
    },
}

/// Convert the output token stream from the `vllm_llm` layer into incrementally
/// decoded text.
#[try_stream]
pub async fn decoded_text_event_stream(
    request_id: String,
    tokenizer: DynTokenizer,
    mut raw_stream: impl Stream<Item = vllm_llm::Result<GenerateOutput>> + Unpin,
    mut decode_options: TextDecodeOptions,
    intermediate: bool,
    mut y: TryYielder<DecodedTextEvent, Error>,
) -> crate::Result<()> {
    let mut decoder: Option<Box<dyn IncrementalDecoder>> = None;
    let mut prompt_token_count = 0_usize;
    let mut cached_token_count = 0_usize;
    let mut token_ids = Vec::new();
    let mut output_token_count: usize = 0;
    let mut logprobs: Option<DecodedLogprobs> = None;

    while let Some(next) = raw_stream.next().await {
        let output = next?;
        cached_token_count = cached_token_count.max(output.cached_token_count);

        // If it's the first output, init states and yield `Start` event.
        if decoder.is_none() {
            let prompt_token_ids =
                output.prompt_token_ids().expect("first llm output must carry prompt token ids");
            prompt_token_count = prompt_token_ids.len();

            let dec = tokenizer.create_decode_stream(
                prompt_token_ids,
                decode_options.skip_special_tokens,
                // If we are excluding stop strings from output, we need to buffer
                // the output so that we don't return the beginning of a stop string
                // when streaming the outputs.
                match decode_options.include_stop_str_in_output {
                    true => 0,
                    false => decode_options
                        .stop_strings
                        .as_ref()
                        .and_then(|stops| stops.iter().map(|ss| ss.len()).max())
                        .unwrap_or(1)
                        .saturating_sub(1),
                },
            );
            decoder = Some(dec);

            y.yield_ok(DecodedTextEvent::Start {
                prompt_token_ids: prompt_token_ids.clone(),
                prompt_logprobs: output
                    .prompt_logprobs()
                    .map(|logprobs| {
                        decode_prompt_logprobs(
                            tokenizer.as_ref(),
                            prompt_token_ids,
                            logprobs,
                            decode_options.skip_special_tokens,
                        )
                    })
                    .transpose()?,
            })
            .await;
        };
        let decoder = decoder.as_mut().unwrap();

        let kv_transfer_params = output.kv_transfer_params;
        let ec_transfer_params = output.ec_transfer_params;
        let mut finish_reason = output.finish_reason;
        let mut stop_str_matched = false;
        let suppress_terminal_stop_token = finish_reason.as_ref().is_some_and(|r| r.is_stop())
            && !decode_options.include_stop_str_in_output;
        let decodable_token_ids = if suppress_terminal_stop_token {
            // Match Python V1 token-stop detokenization by keeping the stop token
            // in metadata while excluding it from user-visible text.
            output.token_ids.split_last().map(|(_, rest)| rest).unwrap_or(&[])
        } else {
            &output.token_ids
        };

        let mut delta: Option<String> = None;
        let mut truncate_output_to = None;
        let mut truncate_tokens_to = None;
        for (tok_idx, &token_id) in decodable_token_ids.iter().enumerate() {
            let new_bytes = decoder.push_token(token_id)?;
            if output_token_count + tok_idx + 1 > decode_options.min_tokens as usize
                && let Some(stops) = decode_options.stop_strings.as_mut()
                && let Some((idx, off)) = matches_stop_string(stops, decoder.output(), new_bytes)
            {
                let stop_str = stops.swap_remove(idx);
                truncate_output_to = match decode_options.include_stop_str_in_output {
                    true => Some(off + stop_str.len()),
                    false => Some(off),
                };
                finish_reason = Some(FinishReason::Stop(Some(StopReason::Text(stop_str))));
                truncate_tokens_to = Some(tok_idx + 1);
                stop_str_matched = true;

                break;
            }

            if intermediate && let Some(chunk) = decoder.next_chunk() {
                if let Some(delta_str) = delta.as_mut() {
                    delta_str.push_str(&chunk);
                } else {
                    delta = Some(chunk);
                }
            }
        }

        let mut new_token_ids = output.token_ids;
        let mut new_logprobs = output.logprobs;

        // Trim tokens and logprobs if we matched stop string.
        if let Some(num_tokens) = truncate_tokens_to {
            new_token_ids.truncate(num_tokens);
            if let Some(logprobs) = &mut new_logprobs {
                logprobs.positions.truncate(num_tokens);
            }
        }

        output_token_count += new_token_ids.len();

        let decoded_logprobs = new_logprobs
            .as_ref()
            .map(|logprobs| {
                decode_logprobs(
                    tokenizer.as_ref(),
                    logprobs,
                    decode_options.skip_special_tokens,
                )
            })
            .transpose()?;

        if !intermediate {
            token_ids.extend(&new_token_ids);
            if let Some(dlp) = decoded_logprobs.as_ref() {
                logprobs
                    .get_or_insert_with(|| DecodedLogprobs { positions: vec![] })
                    .positions
                    .extend_from_slice(&dlp.positions);
            }
        }

        if let Some(reason) = finish_reason {
            // Flush any remaining buffered text.
            let (last_chunk, mut text) = decoder.flush(truncate_output_to)?;
            let text_len = text.len();
            let full_text = tracing::enabled!(Level::TRACE).then(|| text.clone());

            if intermediate {
                if let Some(chunk) = last_chunk {
                    if let Some(delta_str) = delta.as_mut() {
                        delta_str.push_str(&chunk);
                    } else {
                        delta = Some(chunk);
                    }
                }
                token_ids = new_token_ids;
                logprobs = decoded_logprobs;
                text = delta.unwrap_or_default();
            }

            debug!(
                finish_reason = ?reason,
                text_length_bytes = text_len,
                output_token_count = output_token_count,
                "request finished with terminal output"
            );
            if let Some(full_text) = full_text {
                trace!(full_text, "request finished with terminal decoded text");
            }

            // Intentionally drop the stream with explicit cause, so that the engine core
            // can distinguish between such normal completion vs an unexpected
            // early drop.
            if stop_str_matched {
                AbortCause::StopStringMatched.drop_as(raw_stream);
            }

            y.yield_ok(DecodedTextEvent::TextDelta {
                delta: text,
                token_ids,
                logprobs,
                finished: Some(Finished {
                    usage: TokenUsage {
                        prompt_token_count,
                        output_token_count,
                        cached_token_count,
                    },
                    finish_reason: reason,
                    kv_transfer_params,
                    ec_transfer_params,
                }),
            })
            .await;
            return Ok(());
        }

        if intermediate {
            y.yield_ok(DecodedTextEvent::TextDelta {
                delta: delta.unwrap_or_default(),
                token_ids: new_token_ids,
                logprobs: decoded_logprobs,
                finished: None,
            })
            .await;
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })
}

/// If stop string matches, returns tuple
/// (index into stop string vec, byte index of first byte of stop string in
/// output)
///
/// When several stop strings match within the newly generated text (for example
/// when a step appends multiple tokens, or one token decodes to several
/// characters), the stop string that completes earliest in the text is selected,
/// so the result matches appending one character at a time. Ties are broken by
/// stop-list order.
fn matches_stop_string(stops: &[String], output: &str, new_bytes: usize) -> Option<(usize, usize)> {
    // We compare byte subslices to avoid utf8 boundary problem
    let output = output.as_bytes();
    let next_off = (output.len() + 1) - new_bytes;
    stops
        .iter()
        .map(|ss| (ss.as_bytes(), ss.len(), next_off.saturating_sub(ss.len())))
        .enumerate()
        .filter_map(|(ss_idx, (ss, len, start_off))| {
            if len == 0 {
                return None;
            }
            output[start_off..]
                .windows(len)
                .position(|w| w == ss)
                .map(|pos| (ss_idx, start_off + pos, start_off + pos + len))
        })
        // `min_by_key` keeps the first minimum, so ties fall back to stop-list order.
        .min_by_key(|&(_, _, end)| end)
        .map(|(ss_idx, start, _)| (ss_idx, start))
}
