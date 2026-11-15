// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::{Map, Number, Value};
use vllm_tokenizer::DynTokenizer;
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, opt, separated, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::{literal, take_till, take_until};

use super::{Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::reasoning::last_reasoning_boundary;
use crate::tool::{Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{incomplete, parse_buffered_event, partial_prefix_len, safe_text_len_mul};

const REASONING_START: &str = "<|channel>thought\n";
const CHANNEL_START: &str = "<|channel>";
const CHANNEL_END: &str = "<channel|>";
const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

type Gemma4Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq)]
enum Gemma4Event {
    Text { len: usize },
    Reasoning { len: usize },
    ReasoningStart,
    ReasoningEnd,
    ToolCallStart,
    ToolCallHeader { name: String },
    ToolCall { args: Map<String, Value> },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Gemma4ArgsScanState {
    scanned_len: usize,
    in_string: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum Gemma4Mode {
    #[default]
    Text,
    Reasoning,
    Header,
    ToolCall {
        name: String,
        args_scan: Gemma4ArgsScanState,
    },
}

/// Unified parser for Google Gemma4 models.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/main/vllm/parser/gemma4.py>
///
/// Handles Gemma4 reasoning and function-call formats:
///
/// `<|channel>thought\nreasoning<channel|>`
///
/// `<|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>`
///
/// Arguments are emitted only after a full Gemma4 tool call is parsed.
pub struct Gemma4UnifiedParser {
    buffer: String,
    mode: Gemma4Mode,
    emitted_tool_count: usize,
    tokenizer: DynTokenizer,
    channel_start_token_id: u32,
    channel_end_token_id: u32,
}

impl Gemma4UnifiedParser {
    /// Create a Gemma4 parser.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let channel_start_token_id = token_id(tokenizer.as_ref(), CHANNEL_START)?;
        let channel_end_token_id = token_id(tokenizer.as_ref(), CHANNEL_END)?;

        Ok(Self {
            buffer: String::new(),
            mode: Gemma4Mode::default(),
            emitted_tool_count: 0,
            channel_start_token_id,
            channel_end_token_id,
            tokenizer,
        })
    }

    fn apply_event(&mut self, event: Gemma4Event, output: &mut UnifiedParserOutput) -> Result<()> {
        match event {
            Gemma4Event::Text { len: consumed_len } => {
                output.push_text(self.buffer[..consumed_len].to_string());
            }
            Gemma4Event::Reasoning { len: consumed_len } => {
                output.push_reasoning(self.buffer[..consumed_len].to_string());
            }
            Gemma4Event::ReasoningStart => self.mode = Gemma4Mode::Reasoning,
            Gemma4Event::ReasoningEnd => self.mode = Gemma4Mode::Text,
            Gemma4Event::ToolCallStart => self.mode = Gemma4Mode::Header,
            Gemma4Event::ToolCallHeader { name } => {
                self.mode = Gemma4Mode::ToolCall {
                    name,
                    args_scan: Gemma4ArgsScanState::default(),
                };
            }
            Gemma4Event::ToolCall { args } => {
                let mode = std::mem::replace(&mut self.mode, Gemma4Mode::Text);
                let Gemma4Mode::ToolCall { name, .. } = mode else {
                    return Err(parsing_failed!(
                        "Gemma4 arguments without an active tool call"
                    ));
                };
                let arguments = serde_json::to_string(&args)
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

    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        self.mode = match last_reasoning_boundary(
            prompt_token_ids,
            self.channel_start_token_id,
            self.channel_end_token_id,
            self.tokenizer.as_ref(),
        ) {
            Some(true) => Gemma4Mode::Reasoning,
            Some(false) | None => Gemma4Mode::Text,
        };
    }

    fn reset(&mut self) -> String {
        let raw = match std::mem::replace(&mut self.mode, Gemma4Mode::Text) {
            Gemma4Mode::Text => std::mem::take(&mut self.buffer),
            Gemma4Mode::Reasoning => {
                format!("{}{}", REASONING_START, std::mem::take(&mut self.buffer))
            }
            Gemma4Mode::Header => {
                format!("{}{}", TOOL_CALL_START, std::mem::take(&mut self.buffer))
            }
            Gemma4Mode::ToolCall { name, .. } => {
                format!(
                    "{}{}{}{{{}",
                    TOOL_CALL_START,
                    CALL_PREFIX,
                    name,
                    std::mem::take(&mut self.buffer)
                )
            }
        };
        self.mode = Gemma4Mode::Text;
        self.emitted_tool_count = 0;
        raw
    }
}

impl UnifiedParser for Gemma4UnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.emitted_tool_count = 0;
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = {
            parse_buffered_event(&self.buffer, |input| {
                parse_next_gemma4_event(input, &mut self.mode)
            })?
        } {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            Gemma4Mode::Text => output.push_text(std::mem::take(&mut self.buffer)),
            Gemma4Mode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            Gemma4Mode::Header | Gemma4Mode::ToolCall { .. } => {
                return Err(parsing_failed!("incomplete Gemma4 tool call"));
            }
        }

        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Gemma4UnifiedParser::reset(self)
    }
}

/// Parse one Gemma4 event from buffered streaming input.
fn parse_next_gemma4_event(
    input: &mut Gemma4Input<'_>,
    mode: &mut Gemma4Mode,
) -> ModalResult<Gemma4Event> {
    match mode {
        Gemma4Mode::Text => parse_text_event(input),
        Gemma4Mode::Reasoning => parse_reasoning_event(input),
        Gemma4Mode::Header => tool_call_header_event(input),
        Gemma4Mode::ToolCall { args_scan, .. } => tool_call_args_event(input, args_scan),
    }
}

/// Parse a Gemma4 text-mode event.
fn parse_text_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    alt((
        reasoning_start_event,
        tool_call_start_event,
        safe_text_event,
    ))
    .parse_next(input)
}

/// Parse a Gemma4 reasoning-mode event.
fn parse_reasoning_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    alt((
        reasoning_end_event,
        tool_call_start_event,
        safe_reasoning_event,
    ))
    .parse_next(input)
}

/// Parse a Gemma4 reasoning start marker.
fn reasoning_start_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    literal(REASONING_START).value(Gemma4Event::ReasoningStart).parse_next(input)
}

/// Parse a Gemma4 reasoning end marker.
fn reasoning_end_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    literal(CHANNEL_END).value(Gemma4Event::ReasoningEnd).parse_next(input)
}

/// Parse a Gemma4 tool-call start marker.
fn tool_call_start_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    literal(TOOL_CALL_START).value(Gemma4Event::ToolCallStart).parse_next(input)
}

/// Parse a Gemma4 tool-call header.
fn tool_call_header_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    let (name,) = seq!(
        _: literal(CALL_PREFIX),
        gemma4_tool_name,
        _: literal("{"),
    )
    .parse_next(input)?;
    Ok(Gemma4Event::ToolCallHeader { name })
}

/// Parse complete Gemma4 tool-call arguments.
fn tool_call_args_event(
    input: &mut Gemma4Input<'_>,
    args_scan: &mut Gemma4ArgsScanState,
) -> ModalResult<Gemma4Event> {
    let raw_args = gemma4_raw_args_until_tool_call_end(input, args_scan)?;
    let Some(args_input) = raw_args.strip_suffix('}') else {
        return Err(ErrMode::Cut(ContextError::new()));
    };
    let args = parse_gemma4_args(args_input)?;

    Ok(Gemma4Event::ToolCall { args })
}

/// Parse a Gemma4 tool name.
fn gemma4_tool_name(input: &mut Gemma4Input<'_>) -> ModalResult<String> {
    let name = take_until(1.., "{").parse_next(input)?.trim();
    if name.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(name.to_string())
}

/// Parse a safe text run before the next Gemma4 marker.
fn safe_text_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    safe_text_len_mul(input, &[REASONING_START, TOOL_CALL_START])
        .map(|len| Gemma4Event::Text { len })
}

/// Parse a safe reasoning run before the next Gemma4 marker.
fn safe_reasoning_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    safe_text_len_mul(input, &[CHANNEL_END, TOOL_CALL_START])
        .map(|len| Gemma4Event::Reasoning { len })
}

/// Parse raw Gemma4 arguments through the first end marker outside a Gemma string.
fn gemma4_raw_args_until_tool_call_end<'i>(
    input: &mut Gemma4Input<'i>,
    state: &mut Gemma4ArgsScanState,
) -> ModalResult<&'i str> {
    let text = **input;
    if state.scanned_len > text.len() {
        return incomplete();
    }

    loop {
        let rest = &text[state.scanned_len..];
        if state.in_string {
            let Some(string_delim) = rest.find(STRING_DELIM) else {
                state.scanned_len = safe_scan_len(text, state.scanned_len, &[STRING_DELIM]);
                return incomplete();
            };

            state.scanned_len += string_delim + STRING_DELIM.len();
            state.in_string = false;
            continue;
        }

        let next_string_delim = rest.find(STRING_DELIM);
        let next_tool_call_end = rest.find(TOOL_CALL_END);
        match (next_string_delim, next_tool_call_end) {
            (Some(string_delim), Some(tool_call_end)) if tool_call_end < string_delim => {
                let end = state.scanned_len + tool_call_end;
                state.scanned_len = end + TOOL_CALL_END.len();
                input.next_slice(state.scanned_len);
                return Ok(&text[..end]);
            }
            (Some(string_delim), _) => {
                state.scanned_len += string_delim + STRING_DELIM.len();
                state.in_string = true;
            }
            (None, Some(tool_call_end)) => {
                let end = state.scanned_len + tool_call_end;
                state.scanned_len = end + TOOL_CALL_END.len();
                input.next_slice(state.scanned_len);
                return Ok(&text[..end]);
            }
            (None, None) => {
                state.scanned_len =
                    safe_scan_len(text, state.scanned_len, &[STRING_DELIM, TOOL_CALL_END]);
                return incomplete();
            }
        }
    }
}

/// Return the scan length while holding back a split marker prefix.
fn safe_scan_len(text: &str, start: usize, markers: &[&str]) -> usize {
    let max_partial = markers
        .iter()
        .map(|marker| partial_prefix_len(&text[start..], marker))
        .max()
        .unwrap_or(0);
    text.len() - max_partial
}

/// Parse complete Gemma4 custom key-value arguments.
fn parse_gemma4_args(args: &str) -> ModalResult<Map<String, Value>> {
    let mut input = args;
    terminated(gemma4_args, eof).parse_next(&mut input)
}

/// Parse Gemma4's custom key-value argument object content.
fn gemma4_args(input: &mut &str) -> ModalResult<Map<String, Value>> {
    let pairs: Vec<(String, Value)> = delimited(
        ws0,
        terminated(
            separated(0.., gemma4_pair, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)?;
    Ok(pairs.into_iter().collect())
}

/// Parse a Gemma4 key-value pair.
fn gemma4_pair(input: &mut &str) -> ModalResult<(String, Value)> {
    let (key, value) = seq!(
        _: ws0,
        gemma4_key,
        _: ws0,
        _: literal(":"),
        _: ws0,
        gemma4_value,
    )
    .parse_next(input)?;
    Ok((key, value))
}

/// Parse a Gemma4 bare key.
fn gemma4_key(input: &mut &str) -> ModalResult<String> {
    let key = take_till(1.., |char: char| char == ':').parse_next(input)?.trim();
    if key.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(key.to_string())
}

/// Parse a Gemma4 value.
fn gemma4_value(input: &mut &str) -> ModalResult<Value> {
    alt((
        gemma4_string.map(|value: &str| Value::String(value.to_string())),
        gemma4_object.map(Value::Object),
        gemma4_array_value.map(Value::Array),
        gemma4_bare_value,
    ))
    .parse_next(input)
}

/// Parse a Gemma4 string delimited by `<|"|>`.
fn gemma4_string<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited(
        literal(STRING_DELIM),
        take_until(0.., STRING_DELIM),
        literal(STRING_DELIM),
    )
    .parse_next(input)
}

/// Parse a nested Gemma4 object.
fn gemma4_object(input: &mut &str) -> ModalResult<Map<String, Value>> {
    delimited(literal("{"), gemma4_args, literal("}")).parse_next(input)
}

/// Parse a Gemma4 array value.
fn gemma4_array_value(input: &mut &str) -> ModalResult<Vec<Value>> {
    delimited(literal("["), gemma4_array_content, literal("]")).parse_next(input)
}

/// Parse Gemma4 array content.
fn gemma4_array_content(input: &mut &str) -> ModalResult<Vec<Value>> {
    delimited(
        ws0,
        terminated(
            separated(0.., gemma4_value, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)
}

/// Parse a Gemma4 bare scalar.
fn gemma4_bare_value(input: &mut &str) -> ModalResult<Value> {
    take_till(1.., |char: char| matches!(char, ',' | '}' | ']'))
        .map(parse_gemma4_scalar)
        .parse_next(input)
}

/// Parse a Gemma4 comma separator.
fn comma_separator(input: &mut &str) -> ModalResult<()> {
    delimited(ws0, literal(","), ws0).void().parse_next(input)
}

fn parse_gemma4_scalar(value: &str) -> Value {
    let value = value.trim();
    if value.is_empty() {
        return Value::String(String::new());
    }
    if value == "true" {
        return Value::Bool(true);
    }
    if value == "false" {
        return Value::Bool(false);
    }
    if matches!(value, "null" | "none" | "nil" | "NULL" | "None" | "NIL") {
        return Value::Null;
    }
    if value.contains('.') {
        if let Ok(parsed) = value.parse::<f64>()
            && let Some(number) = Number::from_f64(parsed)
        {
            return Value::Number(number);
        }
    } else if let Ok(parsed) = value.parse::<i64>() {
        return Value::Number(Number::from(parsed));
    }

    Value::String(value.to_string())
}
