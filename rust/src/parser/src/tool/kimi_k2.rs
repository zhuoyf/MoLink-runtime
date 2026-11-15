// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use winnow::ascii::{digit1, multispace0 as ws0};
use winnow::combinator::{alt, eof, repeat, seq};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until, take_while};

use super::utils::{JsonObjectScanState, parse_buffered_event, safe_text_len, take_json_object};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::{StructuralTagBuilder, Tool};

const TOOL_CALLS_START: &str = "<|tool_calls_section_begin|>";
const TOOL_CALLS_END: &str = "<|tool_calls_section_end|>";
const TOOL_CALL_START: &str = "<|tool_call_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const TOOL_CALL_ARGUMENT_START: &str = "<|tool_call_argument_begin|>";

type KimiK2Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK2Mode {
    Text,
    ToolBlock,
    Header,
    Arguments { json_scan: JsonObjectScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK2Event {
    Text {
        len: usize,
    },
    ToolCallsStart,
    ToolCallStart,
    ToolCallHeader {
        tool_call_id: String,
        function_name: String,
        function_index: usize,
    },
    Arguments {
        len: usize,
    },
    ToolCallEnd,
    ToolCallsEnd,
    IgnoredRest,
}

/// Tool parser for Kimi K2 token-delimited tool calls.
///
/// Example tool call content:
///
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub struct KimiK2ToolParser {
    buffer: String,
    mode: KimiK2Mode,
    active_tool_index: Option<usize>,
    call_ids: BTreeMap<usize, String>,
}

impl KimiK2ToolParser {
    /// Create a Kimi K2 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: KimiK2Mode::Text,
            active_tool_index: None,
            call_ids: BTreeMap::new(),
        }
    }

    /// Apply one parsed Kimi K2 event to parser state and output.
    fn apply_event(&mut self, event: KimiK2Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            KimiK2Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            KimiK2Event::ToolCallsStart => self.mode = KimiK2Mode::ToolBlock,
            KimiK2Event::ToolCallStart => self.mode = KimiK2Mode::Header,
            KimiK2Event::ToolCallHeader {
                tool_call_id,
                function_name,
                function_index,
            } => {
                let tool_index = function_index;
                self.active_tool_index = Some(tool_index);
                self.mode = KimiK2Mode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                self.call_ids.insert(tool_index, tool_call_id);
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            KimiK2Event::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Kimi K2 arguments without an active tool call"
                    ));
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            KimiK2Event::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = KimiK2Mode::ToolBlock;
            }
            KimiK2Event::ToolCallsEnd => {
                self.active_tool_index = None;
                self.mode = KimiK2Mode::Done;
            }
            KimiK2Event::IgnoredRest => {}
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = KimiK2Mode::Text;
        self.active_tool_index = None;
        self.call_ids.clear();
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for KimiK2ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Kimi.builder())
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.call_ids.get(&tool_index).map(String::as_str)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_kimi_k2_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            KimiK2Mode::Text => output.push_text(&self.buffer),
            KimiK2Mode::ToolBlock | KimiK2Mode::Done => {}
            KimiK2Mode::Header | KimiK2Mode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete Kimi K2 tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        KimiK2ToolParser::reset(self)
    }
}

/// Parse a Kimi K2 event for the current parser mode.
fn parse_next_kimi_k2_event(
    input: &mut KimiK2Input<'_>,
    mode: &mut KimiK2Mode,
) -> ModalResult<KimiK2Event> {
    match mode {
        KimiK2Mode::Text => parse_text_event(input),
        KimiK2Mode::ToolBlock => parse_tool_block_event(input),
        KimiK2Mode::Header => tool_call_header_event(input),
        KimiK2Mode::Arguments { json_scan } => parse_arguments_event(input, json_scan),
        KimiK2Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode Kimi K2 event.
fn parse_text_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse one event inside the Kimi K2 tool-calls section.
fn parse_tool_block_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    alt((tool_calls_end_event, tool_call_start_event)).parse_next(input)
}

/// Parse one event inside a Kimi K2 tool-call arguments payload.
fn parse_arguments_event(
    input: &mut KimiK2Input<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<KimiK2Event> {
    if json_scan.complete() {
        tool_call_end_event(input)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parse a Kimi K2 tool-calls section start marker.
fn tool_calls_start_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    literal(TOOL_CALLS_START).value(KimiK2Event::ToolCallsStart).parse_next(input)
}

/// Parse a Kimi K2 tool-calls section end marker.
fn tool_calls_end_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    (ws0, literal(TOOL_CALLS_END))
        .value(KimiK2Event::ToolCallsEnd)
        .parse_next(input)
}

/// Parse a Kimi K2 tool-call start marker.
fn tool_call_start_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    (ws0, literal(TOOL_CALL_START))
        .value(KimiK2Event::ToolCallStart)
        .parse_next(input)
}

/// Parse a Kimi K2 tool-call end marker.
fn tool_call_end_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    literal(TOOL_CALL_END).value(KimiK2Event::ToolCallEnd).parse_next(input)
}

/// Parse a Kimi K2 tool-call header before the argument marker.
fn tool_call_header_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    let (raw_header, _) = (
        take_until(1.., TOOL_CALL_ARGUMENT_START),
        literal(TOOL_CALL_ARGUMENT_START),
    )
        .parse_next(input)?;

    let tool_call_id = raw_header.trim().to_string();
    let mut header_input = raw_header;
    let (header, _, _) = (tool_header, ws0, eof).parse_next(&mut header_input)?;

    Ok(KimiK2Event::ToolCallHeader {
        tool_call_id,
        function_name: header.function_name,
        function_index: header.function_index,
    })
}

/// Parse a Kimi K2 raw JSON arguments delta.
fn argument_delta_event(
    input: &mut KimiK2Input<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<KimiK2Event> {
    take_json_object(input, json_scan).map(|len| KimiK2Event::Arguments { len })
}

/// Parse a safe text run before the next Kimi K2 tool-calls section.
fn safe_text_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    safe_text_len(input, TOOL_CALLS_START).map(|len| KimiK2Event::Text { len })
}

/// Parse ignored rest after the Kimi K2 tool-calls section ends.
fn ignored_rest_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    rest.value(KimiK2Event::IgnoredRest).parse_next(input)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KimiK2ToolHeader {
    function_name: String,
    function_index: usize,
}

/// Parse a Kimi K2 tool-call header.
fn tool_header(input: &mut &str) -> ModalResult<KimiK2ToolHeader> {
    let (function_name, function_index) = seq!(
        _: ws0,
        _: namespace_prefix,
        tool_name_segment,
        _: literal(":"),
        tool_call_index,
    )
    .parse_next(input)?;

    Ok(KimiK2ToolHeader {
        function_name: function_name.to_string(),
        function_index,
    })
}

/// Parse Kimi K2 namespace segments before the final tool name.
fn namespace_prefix(input: &mut &str) -> ModalResult<()> {
    repeat(0.., namespace_segment).parse_next(input)
}

/// Parse a Kimi K2 namespace segment.
fn namespace_segment<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    let (segment, _) = (tool_name_segment, literal(".")).parse_next(input)?;
    Ok(segment)
}

/// Parse a Kimi K2 tool name segment.
fn tool_name_segment<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    take_while(1.., |ch: char| {
        !ch.is_whitespace() && ch != '<' && ch != ':' && ch != '.'
    })
    .parse_next(input)
}

/// Parse a Kimi K2 tool-call index.
fn tool_call_index(input: &mut &str) -> ModalResult<usize> {
    digit1.parse_to().parse_next(input)
}
