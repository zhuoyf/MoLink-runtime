// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{DeepSeekDsmlToolParser, DsmlTokens};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Tool parser for DeepSeek V4 models.
///
/// Example tool call content:
///
/// ```text
/// <｜DSML｜tool_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜tool_calls>
/// ```
///
/// Arguments are emitted only after a full `invoke` block is parsed.
///
/// V4 reuses the V3.2 DSML invoke/parameter grammar but wraps calls in
/// `<｜DSML｜tool_calls>` instead of `<｜DSML｜function_calls>`.
///
/// DeepSeek V4 relies on DSML markers such as `｜DSML｜`, which are
/// represented as special tokens in the tokenizer and therefore must be
/// preserved during decode for parsing to work.
pub struct DeepSeekV4ToolParser(DeepSeekDsmlToolParser);

impl DeepSeekV4ToolParser {
    /// Create a DeepSeek V4 tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self(DeepSeekDsmlToolParser::new(tools, DsmlTokens::V4))
    }
}

impl ToolParser for DeepSeekV4ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::DeepSeekV4.builder())
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
