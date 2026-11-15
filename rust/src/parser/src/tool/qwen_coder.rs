// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, take_until};

use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::{StructuralTagBuilder, Tool};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

/// Model-specific configuration for the shared Qwen Coder grammar.
///
/// Only the tool-call wrapper tokens vary across models that reuse this
/// grammar; the inner `<function=...>` / `<parameter=...>` tags are always
/// byte-identical. Seed-OSS, for example, wraps the same body in
/// `<seed:tool_call>` / `</seed:tool_call>`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenCoderConfig {
    /// Human-readable parser name used in error messages.
    pub(crate) parser_name: &'static str,
    /// Marker that opens a tool-call block.
    pub(crate) tool_call_start: &'static str,
    /// Marker that closes a tool-call block.
    pub(crate) tool_call_end: &'static str,
}

const QWEN_CODER_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "Qwen Coder",
    tool_call_start: TOOL_CALL_START,
    tool_call_end: TOOL_CALL_END,
};

type QwenCoderInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum QwenCoderMode {
    Text,
    ToolCall { end_marker_scan: MarkerScanState },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QwenCoderEvent {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
}

/// Tool parser for Qwen Coder XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=location>杭州</parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Arguments are emitted only after a full `tool_call` block is parsed.
///
/// Note: parallel calls are represented as repeated
/// `<tool_call>...</tool_call>` blocks, not as multiple calls inside one tag.
pub struct Qwen3CoderToolParser {
    buffer: String,
    mode: QwenCoderMode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    config: QwenCoderConfig,
}

impl Qwen3CoderToolParser {
    /// Create a Qwen Coder tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self::with_config(tools, QWEN_CODER_CONFIG)
    }

    /// Create a parser for a model that reuses the Qwen Coder grammar with a
    /// different tool-call wrapper (e.g. Seed-OSS).
    pub(crate) fn with_config(tools: &[Tool], config: QwenCoderConfig) -> Self {
        Self {
            buffer: String::new(),
            mode: QwenCoderMode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            config,
        }
    }

    /// Apply one parsed Qwen Coder event to parser state and output.
    fn apply_event(&mut self, event: QwenCoderEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            QwenCoderEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            QwenCoderEvent::ToolCallStart => {
                self.mode = QwenCoderMode::ToolCall {
                    end_marker_scan: MarkerScanState::default(),
                };
            }
            QwenCoderEvent::ToolCall { name, raw_params } => {
                self.mode = QwenCoderMode::Text;
                let arguments = self.tool_parameters.convert_params_with_schema(&name, raw_params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = QwenCoderMode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for Qwen3CoderToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Qwen3Coder.builder())
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        let config = self.config;

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_qwen_coder_event(input, &mut self.mode, config)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        if !self.buffer.is_empty() {
            if matches!(self.mode, QwenCoderMode::ToolCall { .. })
                || self.buffer.starts_with(self.config.tool_call_start)
            {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.config.parser_name
                ));
            }
            output.push_text(&self.buffer);
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Qwen3CoderToolParser::reset(self)
    }
}

/// Parse a Qwen Coder event for the current parser mode.
fn parse_next_qwen_coder_event(
    input: &mut QwenCoderInput<'_>,
    mode: &mut QwenCoderMode,
    config: QwenCoderConfig,
) -> ModalResult<QwenCoderEvent> {
    match mode {
        QwenCoderMode::Text => parse_text_event(input, config),
        QwenCoderMode::ToolCall { end_marker_scan } => {
            tool_call_event(input, end_marker_scan, config.tool_call_end)
        }
    }
}

/// Parse a text-mode Qwen Coder event.
fn parse_text_event(
    input: &mut QwenCoderInput<'_>,
    config: QwenCoderConfig,
) -> ModalResult<QwenCoderEvent> {
    alt((
        |input: &mut QwenCoderInput<'_>| tool_call_start_event(input, config.tool_call_start),
        |input: &mut QwenCoderInput<'_>| safe_text_event(input, config.tool_call_start),
    ))
    .parse_next(input)
}

/// Parse a Qwen Coder tool-call start marker.
fn tool_call_start_event(
    input: &mut QwenCoderInput<'_>,
    tool_call_start: &'static str,
) -> ModalResult<QwenCoderEvent> {
    literal(tool_call_start).value(QwenCoderEvent::ToolCallStart).parse_next(input)
}

/// Parse a safe text run before the next Qwen Coder marker.
fn safe_text_event(
    input: &mut QwenCoderInput<'_>,
    tool_call_start: &'static str,
) -> ModalResult<QwenCoderEvent> {
    safe_text_len(input, tool_call_start).map(|len| QwenCoderEvent::Text { len })
}

/// Parse a complete Qwen Coder tool call.
fn tool_call_event(
    input: &mut QwenCoderInput<'_>,
    end_marker_scan: &mut MarkerScanState,
    tool_call_end: &'static str,
) -> ModalResult<QwenCoderEvent> {
    let (body,) = seq!(
        _: ws0,
        take_until_marker(tool_call_end, end_marker_scan),
        _: literal(tool_call_end),
    )
    .parse_next(input)?;

    parse_tool_call_body(body)
}

/// Parse a Qwen Coder function block.
fn function_event(input: &mut &str) -> ModalResult<QwenCoderEvent> {
    let (name, raw_params) = seq!(
        _: literal(FUNCTION_START),
        take_until(1.., ">"),
        _: ">",
        _: ws0,
        repeat(0.., terminated(parameter, ws0)),
        _: literal(FUNCTION_END),
    )
    .parse_next(input)?;

    Ok(QwenCoderEvent::ToolCall {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a Qwen Coder parameter block.
fn parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (name, value) = seq!(
        _: literal(PARAMETER_START),
        take_until(1.., ">"),
        _: ">",
        take_until(0.., PARAMETER_END).map(trim_one_wrapping_newline),
        _: literal(PARAMETER_END),
    )
    .parse_next(input)?;

    Ok((name.to_string(), value.to_string()))
}

/// Parse a Qwen Coder tool-call body.
fn parse_tool_call_body(body: &str) -> ModalResult<QwenCoderEvent> {
    let mut input = body;
    delimited(ws0, function_event, (ws0, eof)).parse_next(&mut input)
}

/// Trim a single leading and trailing newline from a parameter value.
fn trim_one_wrapping_newline(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}
