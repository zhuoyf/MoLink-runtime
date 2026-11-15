// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const PHI4MINI_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Phi4Mini",
    start_marker: "functools[",
    end_marker: "]",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    // Accept both key variants emitted by Phi-4 Mini tool-call templates.
    arguments_key: &["arguments", "parameters"],
};

/// Tool parser for phi-4-mini models.
///
/// Example tool-call content:
///
/// ```text
/// functools[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
/// ```
///
/// phi-4-mini emits an array of tool-call objects wrapped in a `functools[..]`
/// envelope. Each object names the function with `name` and carries its
/// arguments under `arguments` (preferred) or `parameters`. Arguments are
/// already OpenAI-style JSON text, so they are streamed as raw argument deltas
/// without schema conversion or JSON normalization.
pub struct Phi4MiniJsonToolParser {
    inner: JsonToolCallParser,
}

impl Phi4MiniJsonToolParser {
    /// Create a phi-4-mini tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(PHI4MINI_CONFIG),
        }
    }
}

impl ToolParser for Phi4MiniJsonToolParser {
    /// Create a boxed phi-4-mini tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Feed one decoded text chunk through the phi-4-mini parser.
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
