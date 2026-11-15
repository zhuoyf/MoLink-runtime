// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, seq};
use winnow::error::{ContextError, ErrMode};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::{ParamElement, ParamInput, ToolSchemas};
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::Tool;

const NAMESPACE: &str = "]<]minimax[>[";
const TOOL_CALL_START: &str = "]<]minimax[>[<tool_call>";
const TOOL_CALL_END: &str = "]<]minimax[>[</tool_call>";
const INVOKE_START: &str = "]<]minimax[>[<invoke";
const INVOKE_END: &str = "]<]minimax[>[</invoke>";
const ELEMENT_START: &str = "]<]minimax[>[<";
const ELEMENT_END_START: &str = "]<]minimax[>[</";
const MIXED_TEXT_FIELD: &str = "$text";

type MinimaxM3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM3Mode {
    Text,
    ToolBlock { invoke_end_scan: MarkerScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM3Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    Invoke {
        name: String,
        params: Vec<(String, ParamInput)>,
    },
    ToolBlockEnd,
    IgnoredRest,
}

/// Tool parser for MiniMax M3 namespace-delimited XML-style tool calls.
///
/// Example tool call content with recursive parameters:
///
/// ```text
/// ]<]minimax[>[<tool_call>
/// ]<]minimax[>[<invoke name="create_order">
/// ]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
/// ]<]minimax[>[<shipping>
/// ]<]minimax[>[<city>Singapore]<]minimax[>[</city>
/// ]<]minimax[>[<zip>018956]<]minimax[>[</zip>
/// ]<]minimax[>[</shipping>
/// ]<]minimax[>[<items>
/// ]<]minimax[>[<item>
/// ]<]minimax[>[<sku>book-001]<]minimax[>[</sku>
/// ]<]minimax[>[<qty>2]<]minimax[>[</qty>
/// ]<]minimax[>[</item>
/// ]<]minimax[>[</items>
/// ]<]minimax[>[</invoke>
/// ]<]minimax[>[</tool_call>
/// ```
///
/// With a schema where `shipping` is an object and `items` is an array of
/// objects, recursive parameter conversion produces:
///
/// ```json
/// {
///   "user_id": 42,
///   "shipping": {
///     "city": "Singapore",
///     "zip": 18956
///   },
///   "items": [
///     {
///       "sku": "book-001",
///       "qty": 2
///     }
///   ]
/// }
/// ```
///
/// MiniMax M3 emits the namespace marker `]<]minimax[>[` before each structural
/// tag. Arguments are emitted only after a full `<invoke>` block is parsed.
pub struct MinimaxM3ToolParser {
    buffer: String,
    mode: MinimaxM3Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl MinimaxM3ToolParser {
    /// Create a MiniMax M3 tool parser.
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: MinimaxM3Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed MiniMax M3 event to parser state and output.
    fn apply_event(&mut self, event: MinimaxM3Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            MinimaxM3Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            MinimaxM3Event::ToolBlockStart => {
                self.mode = MinimaxM3Mode::ToolBlock {
                    invoke_end_scan: MarkerScanState::default(),
                };
            }
            MinimaxM3Event::Invoke { name, params } => {
                let arguments = self.tool_parameters.convert_params_with_schema(&name, params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
            MinimaxM3Event::ToolBlockEnd => self.mode = MinimaxM3Mode::Done,
            MinimaxM3Event::IgnoredRest => {}
        }
        Ok(())
    }
}

impl ToolParser for MinimaxM3ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_minimax_m3_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            MinimaxM3Mode::Text => {
                output.push_text(&self.buffer);
            }
            MinimaxM3Mode::ToolBlock { .. } => {
                if !self.buffer.trim_start().is_empty() {
                    return Err(parsing_failed!("incomplete MiniMax M3 tool call"));
                }
            }
            MinimaxM3Mode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = MinimaxM3Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a MiniMax M3 event for the current parser mode.
fn parse_next_minimax_m3_event(
    input: &mut MinimaxM3Input<'_>,
    mode: &mut MinimaxM3Mode,
) -> ModalResult<MinimaxM3Event> {
    match mode {
        MinimaxM3Mode::Text => parse_text_event(input),
        MinimaxM3Mode::ToolBlock { invoke_end_scan } => {
            parse_tool_block_event(input, invoke_end_scan)
        }
        MinimaxM3Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode MiniMax M3 event.
fn parse_text_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    alt((tool_block_start_event, safe_text_event)).parse_next(input)
}

/// Parse a MiniMax M3 tool-block start marker.
fn tool_block_start_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    literal(TOOL_CALL_START).value(MinimaxM3Event::ToolBlockStart).parse_next(input)
}

/// Parse a safe text run before the next MiniMax M3 marker.
fn safe_text_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    safe_text_len(input, TOOL_CALL_START).map(|len| MinimaxM3Event::Text { len })
}

/// Parse one event inside a MiniMax M3 tool block.
fn parse_tool_block_event(
    input: &mut MinimaxM3Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM3Event> {
    alt((tool_block_end_event, |input: &mut MinimaxM3Input<'_>| {
        invoke_event(input, invoke_end_scan)
    }))
    .parse_next(input)
}

/// Parse a MiniMax M3 tool-block end marker.
fn tool_block_end_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    (ws0, literal(TOOL_CALL_END))
        .value(MinimaxM3Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a complete MiniMax M3 invoke block.
fn invoke_event(
    input: &mut MinimaxM3Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM3Event> {
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
    let params = parse_invoke_params(body)?;

    Ok(MinimaxM3Event::Invoke {
        name: name.trim().to_string(),
        params,
    })
}

/// Parse all parameter elements inside a complete MiniMax M3 invoke body.
fn parse_invoke_params(invoke_body: &str) -> ModalResult<Vec<(String, ParamInput)>> {
    let mut input = invoke_body;
    let mut elements = Vec::new();

    loop {
        let _ = ws0.parse_next(&mut input)?;
        if input.is_empty() {
            break;
        }
        if input.starts_with(ELEMENT_START) {
            elements.push(parameter_element(&mut input)?);
            continue;
        }
        if input.starts_with(NAMESPACE) {
            return malformed();
        }
        // Be tolerant: ordinary text at an invokeparameter boundary ends this invoke.
        // Keep parsed parameters and drop the remaining invoke body.
        break;
    }

    Ok(elements.into_iter().map(|element| (element.name, element.value)).collect())
}

/// Parse a MiniMax M3 parameter element.
fn parameter_element(input: &mut &str) -> ModalResult<ParamElement> {
    let name = open_element_tag(input)?.to_string();
    let value = element_body(input, &name)?;
    close_element_tag(input, &name)?;
    Ok(ParamElement { name, value })
}

/// Parse a MiniMax M3 opening element tag.
fn open_element_tag<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    let name = seq!(
        _: literal(ELEMENT_START),
        take_until(1.., ">"),
        _: literal(">"),
    )
    .parse_next(input)?;

    let name = name.0;
    if name.starts_with('/') || name.trim().is_empty() {
        return malformed();
    }

    Ok(name)
}

/// Parse a MiniMax M3 closing element tag.
fn close_element_tag(input: &mut &str, name: &str) -> ModalResult<()> {
    literal(ELEMENT_END_START).void().parse_next(input)?;
    literal(name).void().parse_next(input)?;
    literal(">").void().parse_next(input)
}

/// Parse the body of one MiniMax M3 element.
fn element_body(input: &mut &str, closing_name: &str) -> ModalResult<ParamInput> {
    let close_tag = format!("{ELEMENT_END_START}{closing_name}>");
    let mut text = String::new();
    let mut elements = Vec::new();

    loop {
        text.push_str(text_until_namespace(input)?);

        if input.starts_with(&close_tag) {
            // Close tag reached, end of element body.
            break;
        }
        if input.starts_with(ELEMENT_START) {
            // Child element start reached, parse child element recursively.
            elements.push(parameter_element(input)?);
            continue;
        }
        if input.starts_with(NAMESPACE) {
            // Unexpected namespace marker.
            return malformed();
        }
    }

    if elements.is_empty() {
        Ok(ParamInput::Text(text))
    } else {
        if !text.trim().is_empty() {
            push_mixed_text_element(&mut elements, text);
        }
        Ok(ParamInput::Elements(elements))
    }
}

/// Parse text until the next MiniMax M3 namespace marker.
fn text_until_namespace<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    take_until(0.., NAMESPACE).parse_next(input)
}

/// Preserve mixed text content under a reserved object field.
///
/// By default, the field name is `$text`, but if that collides with an existing
/// child element name, prepend `$` until there is no collision.
fn push_mixed_text_element(elements: &mut Vec<ParamElement>, text: String) {
    let mut name = MIXED_TEXT_FIELD.to_string();
    while elements.iter().any(|element| element.name == name) {
        name.insert(0, '$');
    }
    elements.push(ParamElement {
        name,
        value: ParamInput::Text(text),
    });
}

/// Parse a quoted or unquoted XML attribute value from partial streaming input.
fn partial_attr_value<'i>(input: &mut MinimaxM3Input<'i>) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

/// Parse ignored rest after the MiniMax M3 tool block ends.
fn ignored_rest_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    rest.value(MinimaxM3Event::IgnoredRest).parse_next(input)
}

fn malformed<T>() -> ModalResult<T> {
    Err(ErrMode::Cut(ContextError::new()))
}
