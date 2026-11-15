// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::detect_hy_token_suffix;
use crate::reasoning::HyReasoningParser;
use crate::tool::{HyDialect, HyToolMarkers, HyToolParser, StructuralTagBuilder, Tool};
use crate::unified::{CombinedParser, Result, UnifiedParser, UnifiedParserOutput, token_id};

/// Unified reasoning and tool parser for HY3 output.
pub struct HyV3UnifiedParser {
    inner: CombinedParser,
}

impl HyV3UnifiedParser {
    /// Create a HY3 parser using the suffix encoded in tokenizer added tokens.
    pub fn new(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let suffix = detect_hy_token_suffix(tokenizer.as_ref());
        let markers = HyToolMarkers::new(&suffix, HyDialect::V3);
        for marker in markers.iter() {
            token_id(tokenizer.as_ref(), marker)?;
        }

        let reasoning = HyReasoningParser::new(tokenizer, &suffix)?;
        let tool = HyToolParser::new(tools, &suffix, HyDialect::V3);
        Ok(Self {
            inner: CombinedParser::new(Some(Box::new(reasoning)), Some(Box::new(tool))),
        })
    }
}

impl UnifiedParser for HyV3UnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        self.inner.structural_tag_builder()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.inner.parse_into(delta, output)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}
