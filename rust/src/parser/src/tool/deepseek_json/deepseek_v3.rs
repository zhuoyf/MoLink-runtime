// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{DeepSeekJsonFormat, DeepSeekJsonToolParser};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Tool parser for DeepSeek V3 JSON-fenced tool calls.
///
/// Example tool call content:
///
/// ````text
/// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
/// ```json
/// {"location":"Tokyo"}
/// ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
/// ````
///
/// Arguments are already OpenAI-style JSON text inside the markdown fence, so
/// they are streamed as raw argument deltas without schema conversion or JSON
/// normalization.
pub struct DeepSeekV3ToolParser(DeepSeekJsonToolParser);

impl DeepSeekV3ToolParser {
    /// Create a DeepSeek V3 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self(DeepSeekJsonToolParser::new(DeepSeekJsonFormat::V3))
    }
}

impl ToolParser for DeepSeekV3ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::DeepSeekR1.builder())
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.0.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.0.finish()
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }
}
