// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::qwen_coder::{Qwen3CoderToolParser, QwenCoderConfig};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const SEED_OSS_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "Seed-OSS",
    tool_call_start: "<seed:tool_call>",
    tool_call_end: "</seed:tool_call>",
};

/// Tool parser for Seed-OSS XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <seed:tool_call>
/// <function=get_weather>
/// <parameter=location>杭州</parameter>
/// </function>
/// </seed:tool_call>
/// ```
///
/// Seed-OSS shares the Qwen3 Coder grammar exactly; only the two tool-call
/// wrapper tokens differ (`<seed:tool_call>` / `</seed:tool_call>`). The inner
/// `<function=...>` / `<parameter=...>` grammar and schema-driven argument
/// conversion are byte-identical, so this delegates to a
/// [`Qwen3CoderToolParser`] configured with the Seed-OSS markers. This mirrors
/// Python `SeedOssParser(Qwen3Parser)` in `vllm/parser/seed_oss.py`.
///
/// Structured-output tags are intentionally unsupported: the trait default
/// `structural_tag_model() -> None` matches Python `SeedOssEngineToolParser`,
/// which sets `structural_tag_model = None` (xgrammar has no Seed-OSS model).
///
/// The `<seed:tool_call>` wrapper tokens are added-vocabulary tokens
/// (`special = false`), not tokenizer special tokens, so they survive decoding
/// under the production default `skip_special_tokens = true`; the trait default
/// `preserve_special_tokens() == false` is therefore correct, matching the
/// sibling `SeedOssReasoningParser`, which relies on the same for `<seed:think>`.
pub struct SeedOssToolParser {
    inner: Qwen3CoderToolParser,
}

impl SeedOssToolParser {
    /// Create a Seed-OSS tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            inner: Qwen3CoderToolParser::with_config(tools, SEED_OSS_CONFIG),
        }
    }
}

impl ToolParser for SeedOssToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}
