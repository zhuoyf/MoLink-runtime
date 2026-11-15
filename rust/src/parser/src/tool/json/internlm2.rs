// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const INTERNLM2_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "InternLM2",
    start_marker: "<|action_start|><|plugin|>",
    end_marker: "<|action_end|>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    // The Python parser's `get_arguments()` accepts either `parameters` or
    // `arguments` and prefers `parameters` when both are present. This Rust
    // parser uses first-encountered semantics because the header parser only
    // permits one args key per tool-call object; if a future model emits
    // both keys in the same object, the Rust port will accept the first one
    // and reject the trailing one as a syntax error rather than silently
    // shadowing it.
    arguments_key: &["parameters", "arguments"],
};

/// Tool parser for InternLM2 special-token wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <|action_start|><|plugin|>{"name": "get_weather", "parameters": {"location":"Tokyo"}}<|action_end|>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// # Divergences from the Python reference
///
/// This Rust port intentionally diverges from
/// `vllm/tool_parsers/internlm2_tool_parser.py` in two user-visible ways:
///
/// - **Parallel tool calls are supported.** Python silently drops every `<|action_start|>` block
///   after the first (`current_tool_id > 0` returns an empty delta); this parser emits every
///   well-formed block with incrementing `tool_index`. Models that legitimately emit multiple
///   action blocks therefore produce more tool calls under Rust than under Python.
/// - **End-marker bytes inside JSON string values are preserved.** Python does
///   `action.split("<|action_end|>")[0]` which truncates regardless of JSON context; this parser
///   scans matched braces and quotes so a literal `<|action_end|>` inside an arguments string is
///   forwarded intact.
/// - **Only whitespace is allowed before the `{`.** Python's non-streaming
///   `action[action.find("{"):]` drops any bytes before the first `{`, but its streaming path has
///   no equivalent and the model format always emits `<|plugin|>{...`; this parser allows only
///   whitespace there, matching the other JSON parsers in this crate.
/// - **Truncated tool calls error rather than silently dropping.** Python's streaming wrapper
///   swallows mid-stream errors with `except Exception: return None` (logging a traceback) while
///   its non-streaming path raises `JSONDecodeError`; this parser returns an `incomplete InternLM2
///   tool call` error from `finish()`, matching the other JSON parsers and Python's non-streaming
///   behavior.
///
/// # Known unaddressed divergences (TODO)
///
/// The following Python behaviors are NOT yet matched. They are deferred to
/// follow-up work because they require non-local changes to the shared
/// `JsonToolCallParser` core that would affect Hermes / Llama / Mistral /
/// Qwen as well. If a real-world InternLM2 deployment hits one of these,
/// prioritize the corresponding fix.
///
/// - **Arguments value type.** The shared core requires the arguments value to be a JSON object
///   (`take_json_object` rejects anything not starting with `{`). Python's
///   `json.dumps(action_dict.get("parameters", ...))` accepts `null`, arrays, strings, and numbers
///   and round-trips them verbatim. Models that legitimately emit `"parameters":null` will hard-
///   fail under Rust.
/// - **Unknown arguments key.** Python falls back to `{}` via `action_dict.get("parameters",
///   action_dict.get("arguments", {}))` when neither key is present; the Rust header parser raises
///   `parsing failed: invalid InternLM2` for any unrecognized key. A model that emits a typo (e.g.
///   `"params"`) breaks the whole response.
/// - **Field order independence.** The header parser requires the JSON keys to appear in the order
///   `name` then arguments key. Python's `json.loads` + `dict.get` is order-independent, so a model
///   emitting `{"parameters":{...},"name":"foo"}` parses in Python but fails in Rust.
pub struct Internlm2ToolParser {
    inner: JsonToolCallParser,
}

impl Internlm2ToolParser {
    /// Create an InternLM2 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(INTERNLM2_CONFIG),
        }
    }
}

impl ToolParser for Internlm2ToolParser {
    /// Create a boxed InternLM2 tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve special-token markers while decoding, since
    /// `<|action_start|>`, `<|plugin|>`, and `<|action_end|>` are tokenizer
    /// special tokens in InternLM2 models.
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// Feed one decoded text chunk through the InternLM2 parser.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    /// Flush any buffered partial state at end of stream.
    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    /// Clear parser state and return currently uncommitted buffered text.
    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}
