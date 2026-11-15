// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::seq;
use winnow::error::{ModalResult, StrContext};
use winnow::prelude::*;
use winnow::token::literal;

use super::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    argument_delta_event, tool_call_header_event,
};
use crate::tool::utils::{JsonObjectScanState, parse_buffered_event};
use crate::tool::{
    Result, StructuralTagBuilder, Tool, ToolCallDelta, ToolParser, ToolParserOutput,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum LlamaJsonMode {
    Start,
    Header,
    Arguments { json_scan: JsonObjectScanState },
    AfterCall,
    Passthrough,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LlamaJsonEvent {
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallClose,
    Separator,
}

/// Tool parser for strict Llama JSON-template tool calls.
///
/// Example tool call content:
///
/// ```text
/// {"name":"get_weather","parameters":{"location":"Tokyo"}}; {"name":"add","parameters":{"x":1,"y":2}}
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// Natural text at the beginning of the stream permanently disables tool
/// parsing for that assistant output.
pub struct Llama3JsonToolParser {
    buffer: String,
    mode: LlamaJsonMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl Llama3JsonToolParser {
    /// Create a Llama JSON tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: LlamaJsonMode::Start,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Commit the stream to JSON parsing or permanent passthrough.
    fn commit_start(&mut self) -> bool {
        if !matches!(self.mode, LlamaJsonMode::Start) {
            return true;
        }

        if self.buffer.is_empty() {
            return false;
        }

        if self.buffer.starts_with('{') {
            self.mode = LlamaJsonMode::Header;
        } else {
            self.mode = LlamaJsonMode::Passthrough;
        }
        true
    }

    /// Apply one parsed Llama JSON event to parser state and output.
    fn apply_event(&mut self, event: LlamaJsonEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            LlamaJsonEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = LlamaJsonMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            LlamaJsonEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Llama JSON arguments without an active tool call"
                    ));
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            LlamaJsonEvent::ToolCallClose => {
                self.active_tool_index = None;
                self.mode = LlamaJsonMode::AfterCall;
            }
            LlamaJsonEvent::Separator => {
                self.active_tool_index = None;
                self.mode = LlamaJsonMode::Header;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = LlamaJsonMode::Start;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for Llama3JsonToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Llama.builder())
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        if !self.commit_start() {
            return Ok(());
        }

        if matches!(self.mode, LlamaJsonMode::Passthrough) {
            output.push_text(&self.buffer);
            self.buffer.clear();
            return Ok(());
        }

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_llama_json_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            LlamaJsonMode::Start | LlamaJsonMode::Passthrough => {
                output.push_text(&self.buffer);
            }
            LlamaJsonMode::AfterCall if self.buffer.trim().is_empty() => {}
            LlamaJsonMode::Header | LlamaJsonMode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete Llama JSON tool call"));
            }
            LlamaJsonMode::AfterCall => {
                return Err(parsing_failed!("invalid Llama JSON"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Llama3JsonToolParser::reset(self)
    }
}

/// Parse a Llama JSON event for the current parser mode.
fn parse_next_llama_json_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut LlamaJsonMode,
) -> ModalResult<LlamaJsonEvent> {
    match mode {
        LlamaJsonMode::Start | LlamaJsonMode::Passthrough => {
            unreachable!("Llama JSON parser driver must commit before parsing events")
        }
        LlamaJsonMode::Header => llama_tool_call_header_event(input),
        LlamaJsonMode::Arguments { json_scan } => parse_llama_arguments_event(input, json_scan),
        LlamaJsonMode::AfterCall => after_call_event(input),
    }
}

/// Parse a Llama JSON tool-call header.
fn llama_tool_call_header_event(input: &mut JsonToolInput<'_>) -> ModalResult<LlamaJsonEvent> {
    const CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Llama JSON",
        start_marker: "",
        end_marker: "",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: Some(";"),
        name_key: "name",
        arguments_key: &["parameters"],
    };

    match tool_call_header_event(input, CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => {
            Ok(LlamaJsonEvent::ToolCallHeader { function_name })
        }
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Parse one event inside a Llama JSON arguments payload.
fn parse_llama_arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<LlamaJsonEvent> {
    if json_scan.complete() {
        tool_call_close_event(input)
    } else {
        match argument_delta_event(input, json_scan)? {
            JsonToolCallEvent::Arguments { len } => Ok(LlamaJsonEvent::Arguments { len }),
            _ => unreachable!("argument_delta_event only emits Arguments"),
        }
    }
}

/// Parse the outer closing brace for one Llama JSON tool call.
fn tool_call_close_event(input: &mut JsonToolInput<'_>) -> ModalResult<LlamaJsonEvent> {
    seq!(_: ws0, _: literal("}"))
        .value(LlamaJsonEvent::ToolCallClose)
        .parse_next(input)
}

/// Parse a semicolon separator after one Llama JSON tool call.
fn after_call_event(input: &mut JsonToolInput<'_>) -> ModalResult<LlamaJsonEvent> {
    seq!(
        _: ws0,
        _: literal(";"),
        _: ws0,
    )
    .value(LlamaJsonEvent::Separator)
    .context(StrContext::Label("Llama JSON"))
    .parse_next(input)
}
