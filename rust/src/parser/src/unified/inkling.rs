// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::ModalResult;
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::literal;

use vllm_tokenizer::DynTokenizer;

use super::{Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::tool::json::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    tool_call_header_event,
};
use crate::tool::{Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{
    JsonObjectScanState, parse_buffered_event, safe_text_len_mul, take_json_object,
};

const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const CONTENT_INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const CONTENT_TOOL_ERROR: &str = "<|content_tool_error|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const END_MESSAGE: &str = "<|end_message|>";

const IDLE_MARKERS: &[&str] = &[
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TOOL_ERROR,
    END_MESSAGE,
];
const BLOCK_END_MARKERS: &[&str] = &[END_MESSAGE, CONTENT_MODEL_END_SAMPLING];

const INKLING_TOOL_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Inkling",
    start_marker: CONTENT_INVOKE_TOOL_JSON,
    end_marker: END_MESSAGE,
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["args"],
};

type InklingInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum InklingEvent {
    Text { len: usize },
    Reasoning { len: usize },
    TextStart,
    ReasoningStart,
    MessageStart,
    Header { len: usize },
    ToolJsonStart,
    ToolJsonHeader { name: String },
    ToolJsonArgs { len: usize, complete: bool },
    BlockEnd,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum InklingMode {
    #[default]
    Idle,
    MessageHeader {
        pending: String,
    },
    Text,
    Reasoning,
    ToolJsonHeader,
    ToolJsonArgs {
        json_scan: JsonObjectScanState,
    },
    ToolJsonClose,
}

/// Unified parser for Inkling typed content blocks.
pub struct InklingUnifiedParser {
    buffer: String,
    mode: InklingMode,
    emitted_tool_count: usize,
    active_tool_index: Option<usize>,
    tokenizer: DynTokenizer,
    message_model_token_id: u32,
    content_text_token_id: u32,
    content_thinking_token_id: u32,
}

impl InklingUnifiedParser {
    /// Create a Inkling parser.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let message_model_token_id = token_id(tokenizer.as_ref(), MESSAGE_MODEL)?;
        let content_text_token_id = token_id(tokenizer.as_ref(), CONTENT_TEXT)?;
        let content_thinking_token_id = token_id(tokenizer.as_ref(), CONTENT_THINKING)?;

        Ok(Self {
            buffer: String::new(),
            mode: InklingMode::Idle,
            emitted_tool_count: 0,
            active_tool_index: None,
            tokenizer,
            message_model_token_id,
            content_text_token_id,
            content_thinking_token_id,
        })
    }

    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        self.mode = InklingMode::Idle;
        for token_id in prompt_token_ids.iter().rev().copied() {
            if token_id == self.message_model_token_id {
                self.mode = InklingMode::MessageHeader {
                    pending: String::new(),
                };
                return;
            }
            if token_id == self.content_thinking_token_id {
                self.mode = InklingMode::Reasoning;
                return;
            }
            if token_id == self.content_text_token_id {
                self.mode = InklingMode::Text;
                return;
            }
            if self.tokenizer.is_special_id(token_id) {
                return;
            }
        }
    }

    fn apply_event(&mut self, event: InklingEvent, output: &mut UnifiedParserOutput) -> Result<()> {
        match event {
            InklingEvent::Text { len } => output.push_text(self.buffer[..len].to_string()),
            InklingEvent::Reasoning { len } => {
                output.push_reasoning(self.buffer[..len].to_string());
            }
            InklingEvent::MessageStart => {
                self.mode = InklingMode::MessageHeader {
                    pending: String::new(),
                };
            }
            InklingEvent::Header { len } => {
                let InklingMode::MessageHeader { pending } = &mut self.mode else {
                    return Err(parsing_failed!(
                        "Inkling header text outside a message header"
                    ));
                };
                pending.push_str(&self.buffer[..len]);
            }
            InklingEvent::TextStart => self.mode = InklingMode::Text,
            InklingEvent::ReasoningStart => self.mode = InklingMode::Reasoning,
            InklingEvent::ToolJsonStart => self.mode = InklingMode::ToolJsonHeader,
            InklingEvent::ToolJsonHeader { name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = InklingMode::ToolJsonArgs {
                    json_scan: JsonObjectScanState::default(),
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments: String::new(),
                });
            }
            InklingEvent::ToolJsonArgs { len, complete } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Inkling arguments without an active tool call"
                    ));
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..len].to_string(),
                });
                if complete {
                    self.mode = InklingMode::ToolJsonClose;
                }
            }
            InklingEvent::BlockEnd => {
                let mode = std::mem::take(&mut self.mode);
                if let InklingMode::MessageHeader { pending } = mode {
                    output.push_text(pending);
                }
                self.active_tool_index = None;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        let mut uncommitted = match std::mem::take(&mut self.mode) {
            InklingMode::MessageHeader { pending } => pending,
            _ => String::new(),
        };
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        uncommitted.push_str(&std::mem::take(&mut self.buffer));
        uncommitted
    }
}

impl UnifiedParser for InklingUnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.emitted_tool_count = 0;
        self.active_tool_index = None;
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_inkling_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &mut self.mode {
            InklingMode::Idle | InklingMode::Text => {
                output.push_text(std::mem::take(&mut self.buffer))
            }
            InklingMode::MessageHeader { pending } => {
                pending.push_str(&self.buffer);
                self.buffer.clear();
                output.push_text(std::mem::take(pending));
            }
            InklingMode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            InklingMode::ToolJsonHeader
            | InklingMode::ToolJsonArgs { .. }
            | InklingMode::ToolJsonClose => {
                return Err(parsing_failed!("incomplete Inkling tool call"));
            }
        }

        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        InklingUnifiedParser::reset(self)
    }
}

/// Parse one Inkling event from buffered streaming input.
fn parse_next_inkling_event(
    input: &mut InklingInput<'_>,
    mode: &mut InklingMode,
) -> ModalResult<InklingEvent> {
    match mode {
        InklingMode::Idle => parse_idle_event(input),
        InklingMode::MessageHeader { .. } => parse_message_header_event(input),
        InklingMode::Text => parse_text_event(input),
        InklingMode::Reasoning => parse_reasoning_event(input),
        InklingMode::ToolJsonHeader => parse_tool_json_header_event(input),
        InklingMode::ToolJsonArgs { json_scan } => parse_tool_json_args_event(input, json_scan),
        InklingMode::ToolJsonClose => parse_tool_json_close_event(input),
    }
}

/// Parse an event while waiting for a Inkling content kind.
fn parse_idle_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((
        message_start_event,
        reasoning_start_event,
        text_start_event,
        tool_json_start_event,
        raw_text_start_event,
        block_end_event,
        safe_idle_text_event,
    ))
    .parse_next(input)
}

/// Parse a Inkling model-authored message start marker.
fn message_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(MESSAGE_MODEL).value(InklingEvent::MessageStart).parse_next(input)
}

/// Parse an event while waiting for an Inkling message content kind.
fn parse_message_header_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((
        reasoning_start_event,
        text_start_event,
        tool_json_start_event,
        raw_text_start_event,
        block_end_event,
        safe_header_event,
    ))
    .parse_next(input)
}

/// Parse an event inside a Inkling text block.
fn parse_text_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((block_end_event, safe_text_event)).parse_next(input)
}

/// Parse an event inside a Inkling reasoning block.
fn parse_reasoning_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((block_end_event, safe_reasoning_event)).parse_next(input)
}

/// Parse a Inkling text start marker.
fn text_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(CONTENT_TEXT).value(InklingEvent::TextStart).parse_next(input)
}

/// Parse a Inkling reasoning start marker.
fn reasoning_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(CONTENT_THINKING).value(InklingEvent::ReasoningStart).parse_next(input)
}

/// Parse a Inkling JSON tool-call start marker.
fn tool_json_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(CONTENT_INVOKE_TOOL_JSON)
        .value(InklingEvent::ToolJsonStart)
        .parse_next(input)
}

/// Parse a Inkling content kind treated as visible text.
fn raw_text_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((
        literal(CONTENT_INVOKE_TOOL_TEXT),
        literal(CONTENT_TOOL_ERROR),
    ))
    .value(InklingEvent::TextStart)
    .parse_next(input)
}

/// Parse a Inkling block end marker.
fn block_end_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((literal(END_MESSAGE), literal(CONTENT_MODEL_END_SAMPLING)))
        .value(InklingEvent::BlockEnd)
        .parse_next(input)
}

/// Parse safe text while waiting for the next Inkling marker.
fn safe_idle_text_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, IDLE_MARKERS).map(|len| InklingEvent::Text { len })
}

/// Parse safe header text before the next Inkling marker.
fn safe_header_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, IDLE_MARKERS).map(|len| InklingEvent::Header { len })
}

/// Parse safe text before the end of a Inkling text block.
fn safe_text_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, BLOCK_END_MARKERS).map(|len| InklingEvent::Text { len })
}

/// Parse safe reasoning before the end of a Inkling reasoning block.
fn safe_reasoning_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, BLOCK_END_MARKERS).map(|len| InklingEvent::Reasoning { len })
}

/// Parse a Inkling JSON tool-call header.
fn parse_tool_json_header_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    match tool_call_header_event(input, INKLING_TOOL_CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => Ok(InklingEvent::ToolJsonHeader {
            name: function_name,
        }),
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Parse raw Inkling JSON tool-call argument bytes.
fn parse_tool_json_args_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<InklingEvent> {
    let len = take_json_object(input, json_scan)?;
    Ok(InklingEvent::ToolJsonArgs {
        len,
        complete: json_scan.complete(),
    })
}

/// Parse the close of a Inkling JSON tool-call block.
fn parse_tool_json_close_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    seq!(
        _: ws0,
        _: literal("}"),
        _: ws0,
        _: literal(END_MESSAGE),
    )
    .value(InklingEvent::BlockEnd)
    .parse_next(input)
}
