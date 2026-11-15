// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Adapts decoded text updates into parsed assistant deltas.
//!
//! This stage sits between low-level token decoding and final block assembly.
//! It drives one unified parser that may emit normal text, reasoning text, or
//! tool-call deltas, then normalizes those parser events into internal
//! assistant events.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use thiserror_ext::AsReport;
use tracing::warn;
use vllm_parser::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};
use vllm_text::output::DecodedTextEvent;

use crate::Result;
use crate::error::Error;
use crate::event::AssistantBlockKind;
use crate::output::{AssistantEvent, DecodedTextEventStream, generate_tool_call_id};

/// Per-stream unified parsing state.
struct UnifiedParserState {
    /// Parser for the current request stream.
    parser: Box<dyn UnifiedParser>,
    /// Whether unified parsing has already failed for this stream.
    parser_failed: bool,
    /// The parser-local index of the currently open tool call, if any.
    ///
    /// Supported parsers currently emit at most one active tool call at a time.
    /// Change this to an indexed map if a model needs interleaved calls later.
    open_call_index: Option<usize>,
}

impl UnifiedParserState {
    /// Create one fresh unified parsing state for a new streamed response.
    fn new(parser: Box<dyn UnifiedParser>) -> Self {
        Self {
            parser,
            parser_failed: false,
            open_call_index: None,
        }
    }

    /// Initialize parser state once prompt token IDs are available.
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        if self.parser_failed {
            return;
        }

        match self.parser.initialize(prompt_token_ids) {
            Ok(()) => {}
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "failed to initialize unified parser; falling back to plain text deltas"
                );
                self.parser_failed = true;
                self.open_call_index = None;
            }
        }
    }

    /// Convert one decoded text delta into zero or more parsed assistant events.
    fn process_delta(&mut self, delta: String) -> Result<Vec<AssistantEvent>> {
        if self.parser_failed {
            self.open_call_index = None;
            return Ok(text_event(AssistantBlockKind::Text, delta).into_iter().collect());
        }

        let mut output = UnifiedParserOutput::default();
        match self.parser.parse_into(&delta, &mut output) {
            Ok(()) => {
                let mut events = Vec::new();
                self.process_parser_output(output, &mut events)?;
                Ok(events)
            }
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "unified parser failed; falling back to plain text deltas"
                );
                self.parser_failed = true;

                let mut events = Vec::new();
                self.process_parser_output(output, &mut events)?;
                self.open_call_index = None;

                let recovered = self.parser.reset();
                if recovered.is_empty() && events.is_empty() {
                    push_text_delta(&mut events, AssistantBlockKind::Text, delta);
                } else {
                    push_text_delta(&mut events, AssistantBlockKind::Text, recovered);
                }
                Ok(events)
            }
        }
    }

    /// Flush parser state at end-of-stream and close any remaining open calls.
    fn finish(&mut self) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();

        if self.parser_failed {
            return Ok(events);
        }

        match self.parser.finish() {
            Ok(output) => self.process_parser_output(output, &mut events)?,
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "unified parser finish failed; closing open parser state"
                );
                self.parser_failed = true;
                self.open_call_index = None;
                let recovered = self.parser.reset();
                push_text_delta(&mut events, AssistantBlockKind::Text, recovered);
            }
        }

        Ok(events)
    }

    /// Apply one parsed unified output to the current stream state.
    fn process_parser_output(
        &mut self,
        output: UnifiedParserOutput,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        for event in output.events {
            match event {
                UnifiedParserEvent::Text(delta) => {
                    self.open_call_index = None;
                    push_text_delta(events, AssistantBlockKind::Text, delta);
                }
                UnifiedParserEvent::Reasoning(delta) => {
                    self.open_call_index = None;
                    push_text_delta(events, AssistantBlockKind::Reasoning, delta);
                }
                UnifiedParserEvent::ToolCall(item) => {
                    self.process_tool_item(item, events)?;
                }
            }
        }

        Ok(())
    }

    /// Apply one parsed tool-call delta emitted by the parser.
    fn process_tool_item(
        &mut self,
        item: vllm_parser::tool::ToolCallDelta,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        if let Some(name) = item.name {
            let is_new_tool = match self.open_call_index {
                Some(open_call_index) => open_call_index != item.tool_index,
                None => true,
            };
            if is_new_tool {
                let id = self
                    .parser
                    .tool_call_id(item.tool_index)
                    .map(str::to_string)
                    .unwrap_or_else(generate_tool_call_id);
                self.open_call_index = Some(item.tool_index);
                events.push(AssistantEvent::ToolCallStart { id, name });
            }
        }

        if item.arguments.is_empty() {
            return Ok(());
        }
        let Some(open_call_index) = self.open_call_index else {
            return Err(Error::ToolCallStreamInvariant {
                message: format!(
                    "received arguments for tool index {} before any tool-call start",
                    item.tool_index
                ),
            });
        };
        if open_call_index != item.tool_index {
            return Err(Error::ToolCallStreamInvariant {
                message: format!(
                    "received arguments for tool index {} while tool index {} is open",
                    item.tool_index, open_call_index
                ),
            });
        }

        events.push(AssistantEvent::ToolCallArgumentsDelta {
            delta: item.arguments,
        });
        Ok(())
    }
}

/// Build one plain text event if `delta` is non-empty.
fn text_event(kind: AssistantBlockKind, delta: String) -> Option<AssistantEvent> {
    if delta.is_empty() {
        return None;
    }
    Some(AssistantEvent::TextDelta { kind, delta })
}

/// Push one plain text delta if it is non-empty.
fn push_text_delta(events: &mut Vec<AssistantEvent>, kind: AssistantBlockKind, delta: String) {
    if let Some(event) = text_event(kind, delta) {
        events.push(event);
    }
}

/// Wrap one decoded-text stream into the internal unified assistant stream.
#[try_stream]
pub(crate) async fn unified_event_stream(
    decoded_stream: impl DecodedTextEventStream,
    parser: Box<dyn UnifiedParser>,
    mut y: TryYielder<AssistantEvent, Error>,
) -> Result<()> {
    pin_mut!(decoded_stream);

    let mut state = UnifiedParserState::new(parser);

    while let Some(event) = decoded_stream.next().await.transpose()? {
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                state.initialize(&prompt_token_ids);
                y.yield_ok(AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                })
                .await;
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            } => {
                for next in state.process_delta(delta)? {
                    y.yield_ok(next).await;
                }
                if logprobs.is_some() || !token_ids.is_empty() {
                    y.yield_ok(AssistantEvent::LogprobsDelta {
                        logprobs,
                        token_ids,
                    })
                    .await;
                }
                if let Some(finished) = finished {
                    for next in state.finish()? {
                        y.yield_ok(next).await;
                    }
                    y.yield_ok(AssistantEvent::Done {
                        usage: finished.usage,
                        finish_reason: finished.finish_reason,
                        kv_transfer_params: finished.kv_transfer_params,
                        ec_transfer_params: finished.ec_transfer_params,
                    })
                    .await;
                }
            }
        }
    }
    Ok(())
}
