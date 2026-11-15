// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

const QWEN_XML_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Qwen XML",
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    marker_whitespace: JsonToolCallWhitespace::Exact("\n"),
    delimiter: None,
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for Qwen XML-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location":"Tokyo"}}
/// </tool_call>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// Note: parallel calls are represented as repeated
/// `<tool_call>...</tool_call>` blocks, not as multiple calls inside one tag.
pub struct Qwen3XmlToolParser {
    inner: JsonToolCallParser,
}

impl Qwen3XmlToolParser {
    /// Create a Qwen XML tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(QWEN_XML_CONFIG),
        }
    }
}

impl ToolParser for Qwen3XmlToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Qwen3.builder())
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
