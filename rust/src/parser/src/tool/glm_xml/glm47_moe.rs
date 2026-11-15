// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{GlmXmlToolParser, Separator};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Tool parser for GLM-4.7 MoE XML-style tool calls.
///
/// GLM-4.7 reuses the GLM-4.5 parser with a more flexible function-name
/// separator, so the name may be followed by whitespace, a newline, or the
/// first `<arg_key>` tag directly.
pub struct Glm47MoeToolParser(GlmXmlToolParser);

impl Glm47MoeToolParser {
    fn new(tools: &[Tool]) -> Self {
        Self(GlmXmlToolParser::new(tools, Separator::Flexible))
    }
}

impl ToolParser for Glm47MoeToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Glm47.builder())
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
