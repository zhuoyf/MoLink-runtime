// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{DeepSeekDsmlToolParser, DsmlTokens};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Tool parser for DeepSeek V3.2 models.
///
/// Example tool call content:
///
/// ```text
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// ```
///
/// Arguments are emitted only after a full `invoke` block is parsed.
///
/// DeepSeek V3.2 relies on DSML markers such as `｜DSML｜`, which are
/// represented as special tokens in the tokenizer and therefore must be
/// preserved during decode for parsing to work.
pub struct DeepSeekV32ToolParser(DeepSeekDsmlToolParser);

impl DeepSeekV32ToolParser {
    /// Create a DeepSeek V3.2 tool parser.
    pub(super) fn new(tools: &[Tool]) -> Self {
        Self(DeepSeekDsmlToolParser::new(tools, DsmlTokens::V32))
    }
}

impl ToolParser for DeepSeekV32ToolParser {
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
        Some(xgrammar_structural_tag::Model::DeepSeekV32.builder())
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
