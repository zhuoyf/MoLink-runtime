// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser for the Kimi K3 (XTML) chat format.
//!
//! Original Python implementations:
//! - `vllm/reasoning/kimi_k3_reasoning_parser.py`
//! - `vllm/tool_parsers/kimi_k3_tool_parser.py`
//!
//! K3 wraps one assistant message into XTML channels built from the dedicated
//! special tokens `<|open|>`, `<|close|>`, and `<|sep|>`:
//!
//! ```text
//! <|open|>think<|sep|>reasoning<|close|>think<|sep|>
//! <|open|>response<|sep|>visible answer<|close|>response<|sep|>
//! <|open|>tools<|sep|>
//!   <|open|>call tool="get_weather" index="1"<|sep|>
//!     <|open|>argument key="city" type="string"<|sep|>Hangzhou<|close|>argument<|sep|>
//!   <|close|>call<|sep|>
//! <|close|>tools<|sep|>
//! <|close|>message<|sep|>
//! ```
//!
//! In chat serving the generation prompt ends with `<|open|>think<|sep|>`
//! (thinking) or `<|open|>response<|sep|>` (instruct), so the model output
//! starts *inside* that channel without re-emitting the open tag;
//! [`UnifiedParser::initialize`] detects this from the prompt token IDs.
//!
//! Argument decoding mirrors the renderer's type tagging (inverse encoding):
//! `type="string"` values pass the raw text through, other types are
//! JSON-decoded, and a raw `json` block is passed through unmodified. Attribute
//! values reverse the renderer escaping (`&quot;` before `&amp;`).
//!
//! Known limitation (shared with the Python parser): string argument and
//! response bodies are emitted raw, so a value that literally contains
//! `<|close|>argument<|sep|>` or `<|close|>response<|sep|>` is
//! indistinguishable from a real closing marker.

mod structural_tag;

pub use structural_tag::KimiK3StructuralTagBuilder;

use serde_json::{Map, Value};
use vllm_tokenizer::DynTokenizer;
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, preceded, repeat, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_till, take_until, take_while};

use self::structural_tag::KIMI_K3_STRUCTURAL_TAG_BUILDER;
use super::{Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::tool::{StructuralTagBuilder, Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{MarkerScanState, parse_buffered_event, safe_text_len_mul, take_until_marker};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const END_OF_MSG: &str = "<|end_of_msg|>";

const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
const CALL_OPEN: &str = "<|open|>call";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARG_OPEN: &str = "<|open|>argument";
const ARG_CLOSE: &str = "<|close|>argument<|sep|>";
const JSON_OPEN: &str = "<|open|>json";
const JSON_CLOSE: &str = "<|close|>json<|sep|>";

const IDLE_MARKERS: &[&str] = &[
    THINK_OPEN,
    RESPONSE_OPEN,
    TOOLS_OPEN,
    MESSAGE_CLOSE,
    END_OF_MSG,
];
const REASONING_MARKERS: &[&str] = &[THINK_CLOSE, END_OF_MSG];
const RESPONSE_MARKERS: &[&str] = &[RESPONSE_CLOSE, TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG];
const EPILOGUE_MARKERS: &[&str] = &[TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG];
const TOOLS_MARKERS: &[&str] = &[CALL_OPEN, TOOLS_CLOSE, MESSAGE_CLOSE, END_OF_MSG];

/// Channel tags are a couple of text tokens; longer `<|open|>…<|sep|>` spans in
/// the prompt tail (attribute-bearing message opens, message bodies) never name
/// a generation channel.
const MAX_PREFILL_TAG_TOKENS: usize = 8;

type KimiK3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK3Event {
    Text {
        len: usize,
    },
    Reasoning {
        len: usize,
    },
    /// Structural noise consumed without emitting anything.
    Skip,
    ThinkOpen,
    ThinkClose,
    ResponseOpen,
    ResponseClose,
    ToolsOpen,
    ToolsClose,
    /// The assistant message closed; everything after it is ignored.
    MessageEnd,
    CallOpen {
        name: String,
        index: Option<String>,
    },
    CallComplete {
        arguments: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum KimiK3Mode {
    /// Before any channel opens: channel opens are expected, but raw text
    /// falls through as visible text so marker-free output still streams.
    #[default]
    Idle,
    /// Inside the `think` channel.
    Reasoning,
    /// Inside the `response` channel.
    Response,
    /// After the `response` (or `tools`) channel closed: only a `tools`
    /// channel or the message close may follow, and noise is never content.
    Epilogue,
    /// Inside the `tools` channel, between `call` blocks.
    Tools,
    /// Inside one `call` block, buffering its body until the close marker.
    Call {
        name: String,
        index: Option<String>,
        scan: MarkerScanState,
    },
    /// After the message closed: ignore the rest (EOS leakage guard).
    Done,
}

/// Unified parser for Kimi K3 XTML think / response / tools channels.
pub struct KimiK3UnifiedParser {
    buffer: String,
    mode: KimiK3Mode,
    /// Number of calls emitted in the current response.
    emitted_call_count: usize,
    tokenizer: DynTokenizer,
    open_token_id: u32,
    sep_token_id: u32,
}

impl KimiK3UnifiedParser {
    /// Create a Kimi K3 parser.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let open_token_id = token_id(tokenizer.as_ref(), OPEN)?;
        let sep_token_id = token_id(tokenizer.as_ref(), SEP)?;

        Ok(Self {
            buffer: String::new(),
            mode: KimiK3Mode::default(),
            emitted_call_count: 0,
            tokenizer,
            open_token_id,
            sep_token_id,
        })
    }

    /// Detect the prefilled generation channel from the prompt tail.
    ///
    /// `add_generation_prompt` ends the prompt with `<|open|>think<|sep|>` or
    /// `<|open|>response<|sep|>`, so generation starts inside that channel and
    /// never re-emits the open tag. Locate the last `<|open|>…<|sep|>` pair and
    /// decode the short tag between the two structural tokens.
    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        self.mode = KimiK3Mode::Idle;

        let Some(sep_pos) = prompt_token_ids.iter().rposition(|&id| id == self.sep_token_id) else {
            return;
        };
        let Some(open_pos) =
            prompt_token_ids[..sep_pos].iter().rposition(|&id| id == self.open_token_id)
        else {
            return;
        };
        let tag_ids = &prompt_token_ids[open_pos + 1..sep_pos];
        if tag_ids.is_empty() || tag_ids.len() > MAX_PREFILL_TAG_TOKENS {
            return;
        }
        let Ok(tag) = self.tokenizer.decode(tag_ids, /* skip_special_tokens */ false) else {
            return;
        };
        self.mode = match tag.trim() {
            "think" => KimiK3Mode::Reasoning,
            "response" => KimiK3Mode::Response,
            _ => KimiK3Mode::Idle,
        };
    }

    fn apply_event(&mut self, event: KimiK3Event, output: &mut UnifiedParserOutput) -> Result<()> {
        match event {
            KimiK3Event::Text { len } => output.push_text(self.buffer[..len].to_string()),
            KimiK3Event::Reasoning { len } => {
                output.push_reasoning(self.buffer[..len].to_string());
            }
            KimiK3Event::Skip => {}
            KimiK3Event::ThinkOpen => self.mode = KimiK3Mode::Reasoning,
            KimiK3Event::ThinkClose => self.mode = KimiK3Mode::Idle,
            KimiK3Event::ResponseOpen => self.mode = KimiK3Mode::Response,
            KimiK3Event::ResponseClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::ToolsOpen => self.mode = KimiK3Mode::Tools,
            KimiK3Event::ToolsClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::MessageEnd => self.mode = KimiK3Mode::Done,
            KimiK3Event::CallOpen { name, index } => {
                self.mode = KimiK3Mode::Call {
                    name,
                    index,
                    scan: MarkerScanState::default(),
                };
            }
            KimiK3Event::CallComplete { arguments } => {
                let mode = std::mem::replace(&mut self.mode, KimiK3Mode::Tools);
                let KimiK3Mode::Call { name, .. } = mode else {
                    return Err(parsing_failed!(
                        "Kimi K3 call completion without an active tool call"
                    ));
                };
                // An empty/garbage call block without a tool name is dropped,
                // matching the Python parser.
                if name.is_empty() {
                    return Ok(());
                }

                let tool_index = self.emitted_call_count;
                self.emitted_call_count += 1;
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments,
                });
            }
        }
        Ok(())
    }

    fn reset_state(&mut self) -> String {
        self.mode = KimiK3Mode::Idle;
        self.emitted_call_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl UnifiedParser for KimiK3UnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.emitted_call_count = 0;
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(&KIMI_K3_STRUCTURAL_TAG_BUILDER)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_kimi_k3_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            KimiK3Mode::Idle | KimiK3Mode::Response => {
                output.push_text(std::mem::take(&mut self.buffer));
            }
            KimiK3Mode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            KimiK3Mode::Epilogue | KimiK3Mode::Done => self.buffer.clear(),
            // A tools channel truncated between complete calls loses only its
            // closing markers; keep the calls already emitted.
            KimiK3Mode::Tools if self.buffer.is_empty() => {}
            KimiK3Mode::Tools | KimiK3Mode::Call { .. } => {
                return Err(parsing_failed!("incomplete Kimi K3 tool call"));
            }
        }

        self.mode = KimiK3Mode::Idle;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.reset_state()
    }
}

/// Parse one Kimi K3 event from buffered streaming input.
fn parse_next_kimi_k3_event(
    input: &mut KimiK3Input<'_>,
    mode: &mut KimiK3Mode,
) -> ModalResult<KimiK3Event> {
    match mode {
        KimiK3Mode::Idle => parse_idle_event(input),
        KimiK3Mode::Reasoning => parse_reasoning_event(input),
        KimiK3Mode::Response => parse_response_event(input),
        KimiK3Mode::Epilogue => parse_epilogue_event(input),
        KimiK3Mode::Tools => parse_tools_event(input),
        KimiK3Mode::Call { scan, .. } => call_body_event(input, scan),
        KimiK3Mode::Done => parse_done_event(input),
    }
}

/// Parse an event while waiting for the next channel open.
fn parse_idle_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(THINK_OPEN).value(KimiK3Event::ThinkOpen),
        literal(RESPONSE_OPEN).value(KimiK3Event::ResponseOpen),
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_idle_text_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `think` channel.
fn parse_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(THINK_CLOSE).value(KimiK3Event::ThinkClose),
        // `<|end_of_msg|>` can reach the parser under `ignore_eos` or
        // `include_stop_str_in_output`; never leak it into reasoning.
        literal(END_OF_MSG).value(KimiK3Event::MessageEnd),
        safe_reasoning_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `response` channel.
fn parse_response_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(RESPONSE_CLOSE).value(KimiK3Event::ResponseClose),
        // The response body also implicitly ends at a `tools` channel.
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_response_text_event,
    ))
    .parse_next(input)
}

/// Parse an event after the response channel closed.
fn parse_epilogue_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        skip_epilogue_noise_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `tools` channel, between `call` blocks.
fn parse_tools_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        call_open_event,
        literal(TOOLS_CLOSE).value(KimiK3Event::ToolsClose),
        // Defensive: an unterminated tools channel still ends with the message.
        message_end_event,
        skip_tools_noise_event,
    ))
    .parse_next(input)
}

/// Parse a message close or end-of-message marker.
fn message_end_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((literal(MESSAGE_CLOSE), literal(END_OF_MSG)))
        .value(KimiK3Event::MessageEnd)
        .parse_next(input)
}

/// Ignore everything after the assistant message closed.
fn parse_done_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    rest.value(KimiK3Event::Skip).parse_next(input)
}

/// Parse safe text while waiting for the next channel marker.
fn safe_idle_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, IDLE_MARKERS).map(|len| KimiK3Event::Text { len })
}

/// Parse safe reasoning before the think close marker.
fn safe_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, REASONING_MARKERS).map(|len| KimiK3Event::Reasoning { len })
}

/// Parse safe response text before the next channel marker.
fn safe_response_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, RESPONSE_MARKERS).map(|len| KimiK3Event::Text { len })
}

/// Skip non-content noise after the response channel closed.
fn skip_epilogue_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, EPILOGUE_MARKERS).map(|_| KimiK3Event::Skip)
}

/// Skip non-content noise between `call` blocks.
fn skip_tools_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, TOOLS_MARKERS).map(|_| KimiK3Event::Skip)
}

/// Parse a `call` open tag into its tool name and one-based index.
fn call_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let (attrs,) = seq!(
        _: literal(CALL_OPEN),
        take_until(0.., SEP),
        _: literal(SEP),
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    Ok(KimiK3Event::CallOpen {
        name: attr_value(&attrs, "tool").unwrap_or_default().to_string(),
        index: attr_value(&attrs, "index")
            .filter(|index| !index.is_empty())
            .map(str::to_string),
    })
}

/// Parse the buffered call body through `<|close|>call<|sep|>` into a
/// completed call.
fn call_body_event(
    input: &mut KimiK3Input<'_>,
    scan: &mut MarkerScanState,
) -> ModalResult<KimiK3Event> {
    let (body,) = seq!(
        take_until_marker(CALL_CLOSE, scan),
        _: literal(CALL_CLOSE),
    )
    .parse_next(input)?;
    let arguments = parse_call_arguments(body)?;

    Ok(KimiK3Event::CallComplete { arguments })
}

/// Parse a complete `call` body into the OpenAI-style arguments JSON string.
///
/// The body is either one raw `json` block (passed through unmodified) or a
/// sequence of typed `argument` blocks (converted per their `type` tags).
fn parse_call_arguments(body: &str) -> ModalResult<String> {
    let mut input = body;
    terminated(
        delimited(ws0, alt((json_block_arguments, typed_arguments)), ws0),
        eof,
    )
    .parse_next(&mut input)
    .map_err(|_| xtml_error("Kimi K3 call body"))
}

/// Parse one raw `json` argument block, passing its body through unmodified.
fn json_block_arguments(input: &mut &str) -> ModalResult<String> {
    let (raw,) = seq!(
        _: literal(JSON_OPEN),
        _: take_until(0.., SEP), // attrs (`type="object"`), unused on decode
        _: literal(SEP),
        take_until(0.., JSON_CLOSE),
        _: literal(JSON_CLOSE),
    )
    .parse_next(input)?;
    Ok(raw.to_string())
}

/// Parse typed `argument` blocks into a serialized JSON object, preserving
/// argument order.
fn typed_arguments(input: &mut &str) -> ModalResult<String> {
    let pairs: Vec<(String, Value)> =
        repeat(0.., terminated(argument_block, ws0)).parse_next(input)?;
    let arguments = pairs.into_iter().collect::<Map<String, Value>>();
    serde_json::to_string(&arguments).map_err(|_| xtml_error("Kimi K3 arguments"))
}

/// Parse one typed `argument` block into its key/value pair.
fn argument_block(input: &mut &str) -> ModalResult<(String, Value)> {
    let (attrs, raw) = seq!(
        _: literal(ARG_OPEN),
        take_until(0.., SEP),
        _: literal(SEP),
        take_until(0.., ARG_CLOSE),
        _: literal(ARG_CLOSE),
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    let key = attr_value(&attrs, "key").unwrap_or_default().to_string();
    let arg_type = attr_value(&attrs, "type").unwrap_or("string");
    Ok((key, decode_argument_value(arg_type, raw)))
}

/// Decode one typed argument value per its XTML `type` tag.
///
/// `string` values pass the raw text through (the renderer emits them
/// unescaped); other types are JSON-decoded, falling back to the raw text on
/// malformed payloads so one quirky value does not fail the whole call.
fn decode_argument_value(arg_type: &str, raw: &str) -> Value {
    if arg_type == "string" {
        return Value::String(raw.to_string());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Parse a complete XTML attribute string like ` tool="get_weather" index="1"`.
fn parse_tag_attrs(attrs: &str) -> ModalResult<Vec<(String, String)>> {
    let mut input = attrs;
    terminated(repeat(0.., preceded(ws1, tag_attr)), (ws0, eof))
        .parse_next(&mut input)
        .map_err(|_| xtml_error("XTML tag attributes"))
}

/// Parse one XTML `key="value"` attribute pair.
fn tag_attr(input: &mut &str) -> ModalResult<(String, String)> {
    seq!(
        take_while(1.., |char: char| char.is_alphanumeric() || char == '_').map(str::to_string),
        _: literal("=\""),
        take_till(0.., '"').map(unescape_attr_value),
        _: literal("\""),
    )
    .parse_next(input)
}

/// Reverse XTML attribute escaping: `&quot;` first, then `&amp;` (the inverse
/// of the encode order).
fn unescape_attr_value(value: &str) -> String {
    value.replace("&quot;", "\"").replace("&amp;", "&")
}

/// Look up one parsed attribute value by key.
fn attr_value<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(name, _)| name == key).map(|(_, value)| value.as_str())
}

/// Build a cut error for determinably malformed XTML structure.
fn xtml_error(label: &'static str) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    ErrMode::Cut(error)
}
