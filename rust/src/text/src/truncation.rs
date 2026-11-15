// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Which side of the prompt to discard when truncation is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncationSide {
    /// Keep the last N tokens, discarding the prompt prefix.
    Left,
    /// Keep the first N tokens, discarding the prompt suffix.
    Right,
}

/// Maximum number of prompt tokens to retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptTruncationLimit {
    /// Use the input budget remaining after reserving output tokens.
    InputBudget,
    /// Retain at most the given number of prompt tokens.
    Fixed(u64),
}

/// Typed prompt-truncation policy used inside the text frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTruncation {
    /// Maximum number of prompt tokens to retain.
    pub limit: PromptTruncationLimit,
    /// Side from which excess prompt tokens are discarded.
    pub side: TruncationSide,
}

impl PromptTruncation {
    /// Convert the OpenAI-compatible integer representation into a typed policy.
    pub fn from_wire(limit: i64, side: TruncationSide) -> Result<Self> {
        let limit = match limit {
            -1 => PromptTruncationLimit::InputBudget,
            0.. => PromptTruncationLimit::Fixed(limit as u64),
            _ => return Err(Error::InvalidTruncatePromptTokens { value: limit }),
        };
        Ok(Self { limit, side })
    }

    pub(crate) fn apply(self, prompt_token_ids: &mut Vec<u32>, input_budget: u32) -> Result<()> {
        let max_input_tokens = match self.limit {
            PromptTruncationLimit::InputBudget => input_budget,
            PromptTruncationLimit::Fixed(limit) => {
                if limit > u64::from(input_budget) {
                    return Err(Error::TruncatePromptTokensExceedsBudget {
                        value: limit,
                        budget: input_budget,
                    });
                }
                limit as u32
            }
        } as usize;

        if prompt_token_ids.len() <= max_input_tokens {
            return Ok(());
        }

        match self.side {
            TruncationSide::Left => {
                let start = prompt_token_ids.len() - max_input_tokens;
                prompt_token_ids.drain(0..start);
            }
            TruncationSide::Right => prompt_token_ids.truncate(max_input_tokens),
        }
        Ok(())
    }
}
