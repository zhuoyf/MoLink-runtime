// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use trait_set::trait_set;
use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs};

use crate::FinishReason;
use crate::error::{Error, Result};
use crate::event::{AssistantContentBlock, AssistantMessage, ChatEvent};

/// Final structured assistant message plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedAssistantMessage {
    pub message: AssistantMessage,
    pub prompt_token_ids: Arc<[u32]>,
    pub prompt_logprobs: Option<DecodedPromptLogprobs>,
    pub logprobs: Option<DecodedLogprobs>,
    pub token_ids: Vec<u32>,
    pub usage: vllm_llm::TokenUsage,
    pub finish_reason: FinishReason,
    /// Connector-specific KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<serde_json::Value>,
    /// Connector-specific encoder cache transfer parameters for disaggregated
    /// serving.
    pub ec_transfer_params: Option<serde_json::Value>,
}

/// Per-request stream of chat events.
pub struct ChatEventStream {
    request_id: String,
    inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>,
}

impl ChatEventStream {
    pub(crate) fn new(request_id: String, inner: impl crate::output::ChatEventStream) -> Self {
        Self {
            request_id,
            inner: Box::pin(inner),
        }
    }

    /// Return the request ID associated with this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Collect the stream to completion and return the final assembled
    /// assistant message.
    pub async fn collect_message(mut self) -> Result<CollectedAssistantMessage> {
        use futures::StreamExt as _;

        let mut message = AssistantMessage::default();
        let mut prompt_logprobs = None;
        let mut prompt_token_ids: Arc<[u32]> = Arc::from([]);
        let mut logprob_positions: Vec<DecodedPositionLogprobs> = Vec::new();
        let mut token_ids: Vec<u32> = Vec::new();
        while let Some(event) = self.next().await.transpose()? {
            match event {
                ChatEvent::Start {
                    prompt_logprobs: start_prompt_logprobs,
                    prompt_token_ids: start_prompt_token_ids,
                } => {
                    prompt_logprobs = start_prompt_logprobs;
                    prompt_token_ids = start_prompt_token_ids;
                }
                ChatEvent::BlockEnd { block, .. } => message.push_block(block),
                ChatEvent::LogprobsDelta {
                    logprobs,
                    token_ids: delta_ids,
                } => {
                    if let Some(logprobs) = logprobs {
                        logprob_positions.extend(logprobs.positions);
                    }
                    token_ids.extend(delta_ids);
                }
                ChatEvent::Done {
                    message: done,
                    usage,
                    finish_reason,
                    kv_transfer_params,
                    ec_transfer_params,
                } => {
                    return Ok(CollectedAssistantMessage {
                        message: done,
                        prompt_token_ids,
                        prompt_logprobs,
                        logprobs: (!logprob_positions.is_empty()).then_some(DecodedLogprobs {
                            positions: logprob_positions,
                        }),
                        token_ids,
                        usage,
                        finish_reason,
                        kv_transfer_params,
                        ec_transfer_params,
                    });
                }
                ChatEvent::ToolCallEnd { call, .. } => {
                    message.push_block(AssistantContentBlock::ToolCall(call));
                }
                ChatEvent::BlockStart { .. }
                | ChatEvent::BlockDelta { .. }
                | ChatEvent::ToolCallStart { .. }
                | ChatEvent::ToolCallArgumentsDelta { .. } => {}
            }
        }

        // Note: this is actually unreachable, as the underlying stream always emit an
        // error on unexpected close.
        Err(Error::StreamClosedBeforeTerminalOutput {
            request_id: self.request_id,
        })
    }
}

impl Stream for ChatEventStream {
    type Item = Result<ChatEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

trait_set! {
    pub trait ChatEventStreamTrait = Stream<Item = Result<ChatEvent>> + Send + 'static;
}
