// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until, take_while};

use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParserOutput};
use crate::tool::Tool;

mod glm45_moe;
mod glm47_moe;

pub use glm45_moe::Glm45MoeToolParser;
pub use glm47_moe::Glm47MoeToolParser;

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const ARG_KEY_START: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE_START: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

type GlmInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum GlmMode {
    Text,
    ToolCall { tool_call_end_scan: MarkerScanState },
    AfterToolCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Separator {
    /// GLM-4.5/4.6 format: function name must end at a newline before
    /// arguments.
    Newline,
    /// GLM-4.7 format: function name may end at whitespace or directly before
    /// `<arg_key>`.
    Flexible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GlmEvent {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    IgnoredRest,
}

/// Tool parser core for GLM XML-style tool calls.
struct GlmXmlToolParser {
    buffer: String,
    mode: GlmMode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    separator: Separator,
}

impl GlmXmlToolParser {
    /// Create a GLM XML tool parser with a function-name separator.
    fn new(tools: &[Tool], separator: Separator) -> Self {
        Self {
            buffer: String::new(),
            mode: GlmMode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            separator,
        }
    }

    /// Apply one parsed GLM event to parser state and output.
    fn apply_event(&mut self, event: GlmEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            GlmEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            GlmEvent::ToolCallStart => {
                self.mode = GlmMode::ToolCall {
                    tool_call_end_scan: MarkerScanState::default(),
                };
            }
            GlmEvent::ToolCall { name, raw_params } => {
                self.mode = GlmMode::AfterToolCall;
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
            GlmEvent::IgnoredRest => {}
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = GlmMode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_glm_event(input, &mut self.mode, self.separator)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        if !self.buffer.is_empty() {
            match self.mode {
                GlmMode::Text => output.push_text(&self.buffer),
                GlmMode::ToolCall { .. } => {
                    return Err(parsing_failed!("incomplete GLM MoE tool call"));
                }
                GlmMode::AfterToolCall => {}
            }
        }
        let _ = self.reset();
        Ok(output)
    }
}

/// Parse a GLM event for the current parser mode.
fn parse_next_glm_event(
    input: &mut GlmInput<'_>,
    mode: &mut GlmMode,
    separator: Separator,
) -> ModalResult<GlmEvent> {
    match mode {
        GlmMode::Text => parse_text_event(input),
        GlmMode::ToolCall { tool_call_end_scan } => {
            tool_call_event(input, separator, tool_call_end_scan)
        }
        GlmMode::AfterToolCall => after_tool_call_event(input),
    }
}

/// Parse a text-mode GLM event.
fn parse_text_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    alt((tool_call_start_event, safe_text_event)).parse_next(input)
}

/// Parse a GLM tool-call start marker.
fn tool_call_start_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    literal(TOOL_CALL_START).value(GlmEvent::ToolCallStart).parse_next(input)
}

/// Parse a safe text run before the next GLM marker.
fn safe_text_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    safe_text_len(input, TOOL_CALL_START).map(|len| GlmEvent::Text { len })
}

/// Parse text after a completed GLM tool call.
fn after_tool_call_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    ws0.void().parse_next(input)?;
    alt((tool_call_start_event, ignored_rest_event)).parse_next(input)
}

/// Parse a trailing rest after GLM tool calls.
fn ignored_rest_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    rest.value(GlmEvent::IgnoredRest).parse_next(input)
}

/// Parse a complete GLM tool call.
fn tool_call_event(
    input: &mut GlmInput<'_>,
    separator: Separator,
    tool_call_end_scan: &mut MarkerScanState,
) -> ModalResult<GlmEvent> {
    let (body,) = seq!(
        take_until_marker(TOOL_CALL_END, tool_call_end_scan),
        _: literal(TOOL_CALL_END),
    )
    .parse_next(input)?;

    parse_tool_call_body(body, separator)
}

/// Parse a GLM tool-call body.
fn parse_tool_call_body(body: &str, separator: Separator) -> ModalResult<GlmEvent> {
    let mut input = body;
    let (name, raw_params) = match separator {
        Separator::Newline => seq!(
            _: ws0,
            parse_newline_separated_function_name,
            parse_parameters,
            _: ws0,
            _: eof,
        )
        .parse_next(&mut input)?,
        Separator::Flexible => seq!(
            _: ws0,
            parse_flexible_function_name,
            parse_parameters,
            _: ws0,
            _: eof,
        )
        .parse_next(&mut input)?,
    };

    Ok(GlmEvent::ToolCall {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a GLM-4.5 newline-separated function name.
fn parse_newline_separated_function_name<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    terminated(take_until(1.., "\n"), "\n").map(str::trim).parse_next(input)
}

/// Parse a GLM-4.7 whitespace-or-tag-separated function name.
fn parse_flexible_function_name<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    terminated(
        take_while(1.., |ch: char| !ch.is_whitespace() && ch != '<'),
        ws0,
    )
    .parse_next(input)
}

/// Parse GLM argument key-value pairs.
fn parse_parameters(input: &mut &str) -> ModalResult<Vec<(String, String)>> {
    repeat(0.., terminated(parse_parameter, ws0)).parse_next(input)
}

/// Parse a GLM argument key-value pair.
fn parse_parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (key, value) = seq!(
        _: literal(ARG_KEY_START),
        take_until(1.., ARG_KEY_END),
        _: literal(ARG_KEY_END),
        _: ws0,
        _: literal(ARG_VALUE_START),
        take_until(0.., ARG_VALUE_END).map(str::trim),
        _: literal(ARG_VALUE_END),
    )
    .parse_next(input)?;

    Ok((key.trim().to_string(), value.to_string()))
}
