// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::{StructuralTagBuilder, Tool};

const TOOL_CALL_START: &str = "<minimax:tool_call>";
const TOOL_CALL_END: &str = "</minimax:tool_call>";
const INVOKE_START: &str = "<invoke";
const INVOKE_END: &str = "</invoke>";
const PARAMETER_START: &str = "<parameter";
const PARAMETER_END: &str = "</parameter>";

type MinimaxM2Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM2Mode {
    Text,
    ToolBlock { invoke_end_scan: MarkerScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM2Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    Invoke {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    ToolBlockEnd,
    IgnoredRest,
}

/// Tool parser for MiniMax M2 XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <minimax:tool_call><invoke name="get_weather">
/// <parameter name="city">Seattle</parameter>
/// </invoke></minimax:tool_call>
/// ```
///
/// Arguments are emitted only after a full `<invoke>` block is parsed.
pub struct MinimaxM2ToolParser {
    buffer: String,
    mode: MinimaxM2Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl MinimaxM2ToolParser {
    /// Create a MiniMax M2 tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: MinimaxM2Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed MiniMax M2 event to parser state and output.
    fn apply_event(&mut self, event: MinimaxM2Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            MinimaxM2Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            MinimaxM2Event::ToolBlockStart => {
                self.mode = MinimaxM2Mode::ToolBlock {
                    invoke_end_scan: MarkerScanState::default(),
                };
            }
            MinimaxM2Event::Invoke { name, raw_params } => {
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
            MinimaxM2Event::ToolBlockEnd => self.mode = MinimaxM2Mode::Done,
            MinimaxM2Event::IgnoredRest => {}
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = MinimaxM2Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for MinimaxM2ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Minimax.builder())
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_minimax_m2_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            MinimaxM2Mode::Text => {
                output.push_text(&self.buffer);
            }
            MinimaxM2Mode::ToolBlock { .. } => {
                return Err(parsing_failed!("incomplete MiniMax M2 tool call"));
            }
            MinimaxM2Mode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        MinimaxM2ToolParser::reset(self)
    }
}

/// Parse a MiniMax M2 event for the current parser mode.
fn parse_next_minimax_m2_event(
    input: &mut MinimaxM2Input<'_>,
    mode: &mut MinimaxM2Mode,
) -> ModalResult<MinimaxM2Event> {
    match mode {
        MinimaxM2Mode::Text => parse_text_event(input),
        MinimaxM2Mode::ToolBlock { invoke_end_scan } => {
            parse_tool_block_event(input, invoke_end_scan)
        }
        MinimaxM2Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode MiniMax M2 event.
fn parse_text_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    alt((tool_block_start_event, safe_text_event)).parse_next(input)
}

/// Parse a MiniMax M2 tool-block start marker.
fn tool_block_start_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    literal(TOOL_CALL_START).value(MinimaxM2Event::ToolBlockStart).parse_next(input)
}

/// Parse a safe text run before the next MiniMax M2 marker.
fn safe_text_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    safe_text_len(input, TOOL_CALL_START).map(|len| MinimaxM2Event::Text { len })
}

/// Parse one event inside a MiniMax M2 tool block.
fn parse_tool_block_event(
    input: &mut MinimaxM2Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM2Event> {
    alt((tool_block_end_event, |input: &mut MinimaxM2Input<'_>| {
        invoke_event(input, invoke_end_scan)
    }))
    .parse_next(input)
}

/// Parse a MiniMax M2 tool-block end marker.
fn tool_block_end_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    (ws0, literal(TOOL_CALL_END))
        .value(MinimaxM2Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a complete MiniMax M2 invoke block.
fn invoke_event(
    input: &mut MinimaxM2Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM2Event> {
    let (name, body) = seq!(
        _: ws0,
        _: literal(INVOKE_START),
        _: (ws1, literal("name=")),
        partial_attr_value,
        _: literal(">"),
        take_until_marker(INVOKE_END, invoke_end_scan),
        _: literal(INVOKE_END),
    )
    .parse_next(input)?;
    let raw_params = parse_invoke_params(body)?;

    Ok(MinimaxM2Event::Invoke {
        name: name.trim().to_string(),
        raw_params,
    })
}

/// Parse all parameter blocks inside a complete MiniMax M2 invoke body.
fn parse_invoke_params(invoke_body: &str) -> ModalResult<Vec<(String, String)>> {
    let mut input = invoke_body;
    delimited(ws0, repeat(0.., terminated(parameter, ws0)), eof).parse_next(&mut input)
}

/// Parse a MiniMax M2 parameter block.
fn parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (name, value) = seq!(
        _: literal(PARAMETER_START),
        _: (ws1, literal("name=")),
        attr_value,
        _: literal(">"),
        take_until(0.., PARAMETER_END),
        _: literal(PARAMETER_END),
    )
    .parse_next(input)?;

    Ok((name.trim().to_string(), value.to_string()))
}

/// Parse a quoted or unquoted XML attribute value.
fn attr_value<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

/// Parse a quoted or unquoted XML attribute value from partial streaming input.
fn partial_attr_value<'i>(input: &mut MinimaxM2Input<'i>) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

/// Parse ignored rest after the MiniMax M2 tool block ends.
fn ignored_rest_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    rest.value(MinimaxM2Event::IgnoredRest).parse_next(input)
}
