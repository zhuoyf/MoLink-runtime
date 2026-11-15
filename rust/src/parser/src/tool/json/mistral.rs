// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const MISTRAL_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Mistral",
    start_marker: "[TOOL_CALLS] [",
    end_marker: "]",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for Mistral JSON-array tool calls.
///
/// Example tool call content:
///
/// ```text
/// [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location":"Tokyo"}}]
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub struct MistralToolParser {
    inner: JsonToolCallParser,
}

impl MistralToolParser {
    /// Create a Mistral tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(MISTRAL_CONFIG),
        }
    }
}

impl ToolParser for MistralToolParser {
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
