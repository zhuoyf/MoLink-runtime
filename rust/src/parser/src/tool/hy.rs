// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod structural_tag;

use std::sync::Arc;

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use self::structural_tag::HyStructuralTagBuilder;
use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput};
use crate::tool::{StructuralTagBuilder, Tool};

/// Wire-level HY tool-call dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HyDialect {
    V3,
    V4,
}

impl HyDialect {
    fn separator(self) -> &'static str {
        match self {
            Self::V3 => "\n",
            Self::V4 => "",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::V3 => "HY3",
            Self::V4 => "HY4",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HyToolMarkers {
    tool_calls_start: String,
    tool_calls_end: String,
    tool_call_start: String,
    tool_call_end: String,
    tool_sep: Option<String>,
    arg_key_start: String,
    arg_key_end: String,
    arg_value_start: String,
    arg_value_end: String,
}

impl HyToolMarkers {
    pub(crate) fn new(suffix: &str, dialect: HyDialect) -> Self {
        Self {
            tool_calls_start: format!("<tool_calls{suffix}>"),
            tool_calls_end: format!("</tool_calls{suffix}>"),
            tool_call_start: format!("<tool_call{suffix}>"),
            tool_call_end: format!("</tool_call{suffix}>"),
            tool_sep: (dialect == HyDialect::V3).then(|| format!("<tool_sep{suffix}>")),
            arg_key_start: format!("<arg_key{suffix}>"),
            arg_key_end: format!("</arg_key{suffix}>"),
            arg_value_start: format!("<arg_value{suffix}>"),
            arg_value_end: format!("</arg_value{suffix}>"),
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &str> {
        [
            Some(self.tool_calls_start.as_str()),
            Some(self.tool_calls_end.as_str()),
            Some(self.tool_call_start.as_str()),
            Some(self.tool_call_end.as_str()),
            self.tool_sep.as_deref(),
            Some(self.arg_key_start.as_str()),
            Some(self.arg_key_end.as_str()),
            Some(self.arg_value_start.as_str()),
            Some(self.arg_value_end.as_str()),
        ]
        .into_iter()
        .flatten()
    }
}

type HyInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum HyMode {
    Text,
    ToolBlock { tool_call_end_scan: MarkerScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HyEvent {
    Text {
        len: usize,
    },
    ToolBlockStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    ToolBlockEnd,
    IgnoredRest,
}

/// Tool parser for HY XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_calls>
/// <tool_call>get_weather<tool_sep>
/// <arg_key>city</arg_key><arg_value>Beijing</arg_value>
/// </tool_call>
/// </tool_calls>
/// ```
///
/// Arguments are emitted only after a full `<tool_call>` block is parsed.
/// HY marker tokens are added-vocabulary tokens rather than tokenizer special
/// tokens, so the default `preserve_special_tokens() == false` is sufficient.
pub(crate) struct HyToolParser {
    buffer: String,
    mode: HyMode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    dialect: HyDialect,
    markers: Arc<HyToolMarkers>,
    structural_tag_builder: HyStructuralTagBuilder,
}

impl HyToolParser {
    /// Create a HY tool parser for one wire dialect.
    pub(crate) fn new(tools: &[Tool], suffix: &str, dialect: HyDialect) -> Self {
        let markers = Arc::new(HyToolMarkers::new(suffix, dialect));
        Self {
            buffer: String::new(),
            mode: HyMode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            dialect,
            markers: Arc::clone(&markers),
            structural_tag_builder: HyStructuralTagBuilder::new(markers, dialect),
        }
    }

    /// Apply one parsed HY event to parser state and output.
    fn apply_event(&mut self, event: HyEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            HyEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            HyEvent::ToolBlockStart => {
                self.mode = HyMode::ToolBlock {
                    tool_call_end_scan: MarkerScanState::default(),
                };
            }
            HyEvent::ToolCall { name, raw_params } => {
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
            HyEvent::ToolBlockEnd => self.mode = HyMode::Done,
            HyEvent::IgnoredRest => {}
        }
        Ok(())
    }
}

impl ToolParser for HyToolParser {
    // Suffix discovery belongs to the unified HY parser so its reasoning and
    // tool delimiters always use the same tokenizer-derived value.
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Err(ToolParserError::DummyUnifiedParser {
            name: "hy".to_string(),
        })
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(&self.structural_tag_builder)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_hy_event(input, &mut self.mode, &self.markers, self.dialect)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            HyMode::Text => output.push_text(&self.buffer),
            HyMode::ToolBlock { .. } => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.dialect.label()
                ));
            }
            HyMode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = HyMode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a HY event for the current parser mode.
fn parse_next_hy_event(
    input: &mut HyInput<'_>,
    mode: &mut HyMode,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<HyEvent> {
    match mode {
        HyMode::Text => parse_text_event(input, markers),
        HyMode::ToolBlock { tool_call_end_scan } => {
            parse_tool_block_event(input, tool_call_end_scan, markers, dialect)
        }
        HyMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode HY event.
fn parse_text_event(input: &mut HyInput<'_>, markers: &HyToolMarkers) -> ModalResult<HyEvent> {
    alt((
        |input: &mut HyInput<'_>| tool_block_start_event(input, markers),
        |input: &mut HyInput<'_>| safe_text_event(input, markers),
    ))
    .parse_next(input)
}

/// Parse a HY tool-block start marker.
fn tool_block_start_event(
    input: &mut HyInput<'_>,
    markers: &HyToolMarkers,
) -> ModalResult<HyEvent> {
    literal(markers.tool_calls_start.as_str())
        .value(HyEvent::ToolBlockStart)
        .parse_next(input)
}

/// Parse a safe text run before the next HY marker.
fn safe_text_event(input: &mut HyInput<'_>, markers: &HyToolMarkers) -> ModalResult<HyEvent> {
    safe_text_len(input, &markers.tool_calls_start).map(|len| HyEvent::Text { len })
}

/// Parse one event inside a HY tool block.
fn parse_tool_block_event(
    input: &mut HyInput<'_>,
    tool_call_end_scan: &mut MarkerScanState,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<HyEvent> {
    alt((
        |input: &mut HyInput<'_>| tool_block_end_event(input, markers),
        |input: &mut HyInput<'_>| tool_call_event(input, tool_call_end_scan, markers, dialect),
    ))
    .parse_next(input)
}

/// Parse a HY tool-block end marker.
fn tool_block_end_event(input: &mut HyInput<'_>, markers: &HyToolMarkers) -> ModalResult<HyEvent> {
    (ws0, literal(markers.tool_calls_end.as_str()))
        .value(HyEvent::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a complete HY tool-call block.
fn tool_call_event(
    input: &mut HyInput<'_>,
    tool_call_end_scan: &mut MarkerScanState,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<HyEvent> {
    let (body,) = seq!(
        _: ws0,
        _: literal(markers.tool_call_start.as_str()),
        take_until_marker(markers.tool_call_end.as_str(), tool_call_end_scan),
        _: literal(markers.tool_call_end.as_str()),
    )
    .parse_next(input)?;
    let mut body_input = body;
    let (name, params) = parse_tool_call_body(&mut body_input, markers, dialect)?;
    let raw_params = parse_tool_call_params(params, markers)?;

    Ok(HyEvent::ToolCall {
        name: name.trim().to_string(),
        raw_params,
    })
}

/// Parse a complete HY tool-call body according to its dialect.
fn parse_tool_call_body<'i>(
    input: &mut &'i str,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<(&'i str, &'i str)> {
    match dialect {
        HyDialect::V3 => {
            let tool_sep = markers.tool_sep.as_deref().expect("HY3 has a tool separator");
            terminated(
                seq!(
                    take_until(0.., tool_sep),
                    _: literal(tool_sep),
                    rest,
                ),
                eof,
            )
            .parse_next(input)
        }
        HyDialect::V4 => terminated(
            alt((
                seq!(take_until(0.., markers.arg_key_start.as_str()), rest,),
                rest.map(|name| (name, "")),
            )),
            eof,
        )
        .parse_next(input),
    }
}

/// Parse all parameter blocks inside a complete HY tool call.
fn parse_tool_call_params(
    tool_call_body: &str,
    markers: &HyToolMarkers,
) -> ModalResult<Vec<(String, String)>> {
    let mut input = tool_call_body;
    delimited(
        ws0,
        repeat(
            0..,
            terminated(|input: &mut &str| parameter(input, markers), ws0),
        ),
        eof,
    )
    .parse_next(&mut input)
}

/// Parse a HY argument key/value block.
fn parameter(input: &mut &str, markers: &HyToolMarkers) -> ModalResult<(String, String)> {
    let (name, value) = seq!(
        _: literal(markers.arg_key_start.as_str()),
        take_until(0.., markers.arg_key_end.as_str()),
        _: literal(markers.arg_key_end.as_str()),
        _: ws0,
        _: literal(markers.arg_value_start.as_str()),
        take_until(0.., markers.arg_value_end.as_str()),
        _: literal(markers.arg_value_end.as_str()),
    )
    .parse_next(input)?;

    Ok((name.trim().to_string(), value.to_string()))
}

/// Parse ignored rest after the HY tool block ends.
fn ignored_rest_event(input: &mut HyInput<'_>) -> ModalResult<HyEvent> {
    rest.value(HyEvent::IgnoredRest).parse_next(input)
}
