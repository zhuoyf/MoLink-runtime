// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, peek, seq};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::token::{any, literal};

use super::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    tool_call_header_event,
};
use crate::tool::utils::{
    JsonObjectScanState, JsonStringScanState, decode_json_str, parse_buffered_event, safe_text_len,
    take_json_object, take_json_string,
};
use crate::tool::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite4Mode {
    Text,
    Header,
    /// Parsing the arguments value:
    /// `None` until the first byte decides object vs string;
    /// `Some` while streaming the selected value shape.
    Args {
        args_scan: Option<Granite4ArgsScan>,
    },
    /// Arguments done; consume the object's closing `}` and `</tool_call>`.
    Close,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite4ArgsScan {
    Object(JsonObjectScanState),
    String(JsonStringScanState),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite4Event {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCallHeader {
        function_name: String,
    },
    /// Verbatim bytes of an object-valued arguments payload; `complete` once the
    /// object scan reaches its closing brace.
    ObjectArgsDelta {
        len: usize,
        complete: bool,
    },
    /// Decoded contents of a string-valued arguments payload.
    StringArgs {
        decoded: String,
    },
    ToolCallEnd,
}

/// Tool parser for Granite 4 `<tool_call>…</tool_call>` JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>{"name": "get_weather", "arguments": {"city": "Boston"}}</tool_call>
/// ```
///
/// Parallel calls are repeated `<tool_call>…</tool_call>` blocks with ordinary
/// content interleaved between them. This reuses the shared JSON helpers for
/// everything except one Granite 4 specific step (`args_event`): the `arguments`
/// value may be a JSON object (kept verbatim) **or** a JSON string whose decoded
/// contents are the arguments (the `# test granite behavior` case in Python).
pub struct Granite4ToolParser {
    buffer: String,
    mode: Granite4Mode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl Granite4ToolParser {
    /// Create a Granite 4 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Granite4Mode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Apply one parsed Granite 4 event to parser state and output.
    fn apply_event(&mut self, event: Granite4Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            Granite4Event::Text { len } => output.push_text(&self.buffer[..len]),
            Granite4Event::ToolCallStart => self.mode = Granite4Mode::Header,
            Granite4Event::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = Granite4Mode::Args { args_scan: None };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            Granite4Event::ObjectArgsDelta { len, complete } => {
                let arguments = self.buffer[..len].to_string();
                self.push_arguments(arguments, output)?;
                if complete {
                    self.mode = Granite4Mode::Close;
                }
            }
            Granite4Event::StringArgs { decoded } => {
                self.push_arguments(decoded, output)?;
                self.mode = Granite4Mode::Close;
            }
            Granite4Event::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = Granite4Mode::Text;
            }
        }
        Ok(())
    }

    /// Append one arguments delta to the active tool call.
    fn push_arguments(&self, arguments: String, output: &mut ToolParserOutput) -> Result<()> {
        let Some(tool_index) = self.active_tool_index else {
            return Err(parsing_failed!(
                "Granite4 arguments without an active tool call"
            ));
        };
        output.push_call(ToolCallDelta {
            tool_index,
            name: None,
            arguments,
        });
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = Granite4Mode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for Granite4ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_granite4_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            Granite4Mode::Text => output.push_text(&self.buffer),
            Granite4Mode::Header | Granite4Mode::Args { .. } | Granite4Mode::Close => {
                return Err(parsing_failed!("incomplete Granite4 tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Granite4ToolParser::reset(self)
    }
}

/// Parse a Granite 4 event for the current parser mode.
fn parse_next_granite4_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut Granite4Mode,
) -> ModalResult<Granite4Event> {
    match mode {
        Granite4Mode::Text => text_event(input),
        Granite4Mode::Header => header_event(input),
        Granite4Mode::Args { args_scan } => args_event(input, args_scan),
        Granite4Mode::Close => close_event(input),
    }
}

/// Parse content text or the start of a `<tool_call>` block. *(reuses `safe_text_len`)*
fn text_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite4Event> {
    alt((
        |input: &mut JsonToolInput<'_>| {
            seq!(_: literal(TOOL_CALL_START), _: ws0)
                .value(Granite4Event::ToolCallStart)
                .parse_next(input)
        },
        |input: &mut JsonToolInput<'_>| {
            safe_text_len(input, TOOL_CALL_START).map(|len| Granite4Event::Text { len })
        },
    ))
    .parse_next(input)
}

/// Parse the `{"name":"X","arguments":` header before the value. *(reuses `tool_call_header_event`)*
fn header_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite4Event> {
    const CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Granite4",
        start_marker: "",
        end_marker: "",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: None,
        name_key: "name",
        arguments_key: &["arguments"],
    };

    match tool_call_header_event(input, CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => {
            Ok(Granite4Event::ToolCallHeader { function_name })
        }
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Parse one arguments-value event.
///
/// GRANITE 4 SPECIFIC - the sole behavior that differs from the shared
/// `<tool_call>` JSON parsers. The value is either a JSON object (kept verbatim,
/// streamed incrementally via `take_json_object`) or an escaped JSON string
/// (decoded whole via `json_str`). The string form is why we cannot just forward
/// raw arg bytes like the sibling parsers do: an escaped string only resolves
/// once seen whole and unescaped.
fn args_event(
    input: &mut JsonToolInput<'_>,
    args_scan: &mut Option<Granite4ArgsScan>,
) -> ModalResult<Granite4Event> {
    if let Some(scan) = args_scan {
        return match scan {
            Granite4ArgsScan::Object(scan) => {
                let len = take_json_object(input, scan)?;
                Ok(Granite4Event::ObjectArgsDelta {
                    len,
                    complete: scan.complete(),
                })
            }
            Granite4ArgsScan::String(scan) => string_args_event(input, scan),
        };
    }

    match peek(any).parse_next(input)? {
        '{' => {
            let mut scan = JsonObjectScanState::default();
            let len = take_json_object(input, &mut scan)?;
            let complete = scan.complete();
            *args_scan = Some(Granite4ArgsScan::Object(scan));
            Ok(Granite4Event::ObjectArgsDelta { len, complete })
        }
        '"' => {
            *args_scan = Some(Granite4ArgsScan::String(JsonStringScanState::default()));
            let Some(Granite4ArgsScan::String(scan)) = args_scan else {
                unreachable!("Granite4 string scan state was just initialized")
            };
            string_args_event(input, scan)
        }
        _ => {
            let mut error = ContextError::new();
            error.push(StrContext::Label("Granite4 arguments"));
            Err(ErrMode::Cut(error))
        }
    }
}

fn string_args_event(
    input: &mut JsonToolInput<'_>,
    scan: &mut JsonStringScanState,
) -> ModalResult<Granite4Event> {
    let text = **input;
    let len = take_json_string(input, scan)?;
    Ok(Granite4Event::StringArgs {
        decoded: decode_json_str(&text[..len])?,
    })
}

/// Parse the tool-call object's closing `}` and the `</tool_call>` end marker.
fn close_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite4Event> {
    seq!(_: ws0, _: literal("}"), _: ws0, _: literal(TOOL_CALL_END))
        .value(Granite4Event::ToolCallEnd)
        .parse_next(input)
}
