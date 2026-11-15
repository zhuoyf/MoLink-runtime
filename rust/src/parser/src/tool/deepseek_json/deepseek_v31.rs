// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{DeepSeekJsonFormat, DeepSeekJsonToolParser};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Tool parser for DeepSeek V3.1 raw JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub struct DeepSeekV31ToolParser(DeepSeekJsonToolParser);

impl DeepSeekV31ToolParser {
    /// Create a DeepSeek V3.1 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self(DeepSeekJsonToolParser::new(DeepSeekJsonFormat::V31))
    }
}

impl ToolParser for DeepSeekV31ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::DeepSeekV31.builder())
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
