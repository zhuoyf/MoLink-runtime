// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Adapter that combines reasoning and tool parsers.

use vllm_tokenizer::DynTokenizer;

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput};
use crate::reasoning::ReasoningParser;
use crate::tool::{StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Unified parser that composes existing reasoning and tool parsers.
pub struct CombinedParser {
    reasoning: Option<Box<dyn ReasoningParser>>,
    tool: Option<Box<dyn ToolParser>>,
}

impl CombinedParser {
    /// Create a combined parser from optional reasoning and tool parsers.
    pub fn new(
        reasoning: Option<Box<dyn ReasoningParser>>,
        tool: Option<Box<dyn ToolParser>>,
    ) -> Self {
        Self { reasoning, tool }
    }

    /// Create a text-only combined parser.
    pub fn plain_text_only() -> Self {
        Self {
            reasoning: None,
            tool: None,
        }
    }

    fn parse_tool(&mut self, content: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        let Some(tool) = self.tool.as_mut() else {
            output.push_text(content);
            return Ok(());
        };

        // Preserve any tool output that was already produced before the error.
        let mut tool_output = ToolParserOutput::default();
        let result = tool.parse_into(content, &mut tool_output);
        output.append_tool_output(tool_output);
        result?;

        Ok(())
    }

    fn flush_tool(&mut self) -> Result<UnifiedParserOutput> {
        let Some(tool) = self.tool.as_mut() else {
            return Ok(UnifiedParserOutput::default());
        };

        let output = tool.finish()?;
        let mut unified = UnifiedParserOutput::default();
        unified.append_tool_output(output);
        Ok(unified)
    }
}

impl UnifiedParser for CombinedParser {
    fn create(_tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Err(UnifiedParserError::CombinedParserConstructor)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        if let Some(reasoning) = self.reasoning.as_mut() {
            reasoning.initialize(prompt_token_ids)?;
        }
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        self.reasoning.as_ref().is_some_and(|parser| parser.preserve_special_tokens())
            || self.tool.as_ref().is_some_and(|parser| parser.preserve_special_tokens())
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        self.tool.as_ref().and_then(|parser| parser.structural_tag_builder())
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.tool.as_ref().and_then(|parser| parser.tool_call_id(tool_index))
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        let Some(reasoning) = self.reasoning.as_mut() else {
            return self.parse_tool(delta, output);
        };

        let reasoning_delta = reasoning.push(delta)?;
        if let Some(reasoning) = reasoning_delta.reasoning {
            output.push_reasoning(reasoning);
        }
        if let Some(content) = reasoning_delta.content {
            self.parse_tool(&content, output)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        if let Some(reasoning) = self.reasoning.as_mut() {
            let reasoning_delta = reasoning.finish()?;
            if let Some(reasoning) = reasoning_delta.reasoning {
                output.push_reasoning(reasoning);
            }
            if let Some(content) = reasoning_delta.content {
                self.parse_tool(&content, &mut output)?;
            }
        }
        output.append(self.flush_tool()?);
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.tool.as_mut().map_or_else(String::new, |parser| parser.reset())
    }
}
