// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::mem::take;

use crate::{Result, Tokenizer};

/// Stateful incremental decoder that emits text chunks one token at a time.
pub trait IncrementalDecoder: Send {
    /// Push one generated token and return how many new string bytes were
    /// added.
    fn push_token(&mut self, token_id: u32) -> Result<usize>;

    /// Consume any text which is currently ready.
    fn next_chunk(&mut self) -> Option<String>;

    /// Flush any remaining buffered text that has not yet been emitted.
    ///
    /// Called after the final generated token to force out buffered/incomplete
    /// fragments.
    fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)>;

    /// Return cumulative decoded text so far.
    fn output(&self) -> &str;
}

/// [`IncrementalDecoder`] built on [`Tokenizer::decode()`] with prefix-diffing.
///
/// This is the same sliding-window algorithm used by `tokenizers::DecodeStream`
pub(crate) struct DecodeStream<'a, T: Tokenizer + ?Sized> {
    tokenizer: &'a T,
    skip_special_tokens: bool,
    min_bytes_to_buffer: usize,
    // mutated state
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    prefix_seeded: bool,
    cumulative_output: String,
    output_index: usize,
}

impl<'a, T: Tokenizer + ?Sized> DecodeStream<'a, T> {
    pub(crate) fn new(
        tokenizer: &'a T,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            min_bytes_to_buffer,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
            prefix_seeded: prompt_token_ids.is_empty(),
            cumulative_output: String::new(),
            output_index: 0,
        }
    }
}

/// Try a short tail suffix first (covers a CJK glyph straddling 1-2 token
/// boundaries); beyond 6 tokens the fallback full-prompt decode is no worse
/// than baseline so widening the sweep just adds overhead.
const SAFE_SUFFIX_MIN: usize = 4;
const SAFE_SUFFIX_MAX: usize = 6;

impl<T: Tokenizer + ?Sized> DecodeStream<'_, T> {
    /// Decode prompt-only context for prefix seeding.
    ///
    /// Prompt ids may come from the model vocabulary rather than the local
    /// tokenizer vocabulary. For this seeding path, ids that cannot be mapped
    /// back to raw token text are dropped before retrying strict decode. This
    /// tolerance is intentionally limited to prompt context; generated ids are
    /// decoded later through the normal strict path.
    fn decode_prompt_context(&self, ids: &[u32]) -> Result<(String, Vec<u32>)> {
        match self.tokenizer.decode(ids, self.skip_special_tokens) {
            Ok(decoded) => Ok((decoded, ids.to_vec())),
            Err(error) => {
                let filtered = ids
                    .iter()
                    .copied()
                    .filter(|&id| self.tokenizer.id_to_token(id).is_some())
                    .collect::<Vec<_>>();
                if filtered.len() == ids.len() {
                    return Err(error);
                }
                self.tokenizer
                    .decode(&filtered, self.skip_special_tokens)
                    .map(|decoded| (decoded, filtered))
            }
        }
    }

    /// Seed `self.prefix` from the shortest trailing prompt suffix whose
    /// filtered context is still long enough and whose decoded text has no
    /// U+FFFD. A clean decode means the suffix starts and ends at valid
    /// UTF-8/token boundaries, so priming from it is equivalent to priming from
    /// the full prompt.
    fn seed_prefix(&mut self) -> Result<()> {
        let prompt_len = self.ids.len();
        if prompt_len > SAFE_SUFFIX_MIN {
            let max_try = SAFE_SUFFIX_MAX.min(prompt_len - 1);
            for suffix_len in SAFE_SUFFIX_MIN..=max_try {
                let start = prompt_len - suffix_len;
                let (decoded, context_ids) = self.decode_prompt_context(&self.ids[start..])?;
                if !decoded.contains('\u{FFFD}') && context_ids.len() >= SAFE_SUFFIX_MIN {
                    self.prefix = decoded;
                    self.ids = context_ids;
                    self.prefix_index = self.ids.len();
                    return Ok(());
                }
            }
        }
        let (decoded, context_ids) = self.decode_prompt_context(&self.ids)?;
        self.ids = context_ids;
        if !decoded.ends_with('\u{FFFD}') {
            self.prefix = decoded;
            self.prefix_index = self.ids.len();
        }
        Ok(())
    }
}

impl<T: Tokenizer + ?Sized> IncrementalDecoder for DecodeStream<'_, T> {
    fn push_token(&mut self, token_id: u32) -> Result<usize> {
        if !self.prefix_seeded && !self.ids.is_empty() {
            self.seed_prefix()?;
            self.prefix_seeded = true;
        }

        self.ids.push(token_id);
        let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        let prefix_len = self.prefix.len();
        if string.len() <= prefix_len || string.ends_with('\u{FFFD}') {
            return Ok(0);
        }
        // Ensure we split at a utf-8 char boundary.
        let new_chunk = &string[string.floor_char_boundary(prefix_len)..];
        self.cumulative_output.push_str(new_chunk);
        self.ids.drain(..self.prefix_index);
        self.prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        self.prefix_index = self.ids.len();
        Ok(new_chunk.len())
    }

    fn next_chunk(&mut self) -> Option<String> {
        let cutoff = self.cumulative_output.len().saturating_sub(self.min_bytes_to_buffer);
        // Ensure we split at a utf-8 char boundary.
        let cutoff = self.cumulative_output.floor_char_boundary(cutoff);
        (cutoff > self.output_index).then(|| {
            let chunk = self.cumulative_output[self.output_index..cutoff].to_string();
            self.output_index = cutoff;
            chunk
        })
    }

    fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)> {
        // If the prefix was never seeded (no push_token was called), `ids`
        // holds only prompt context — decoding it would re-emit prompt text.
        if self.prefix_seeded && !self.ids.is_empty() {
            let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            let prefix_len = self.prefix.len();
            // Ensure we split at a utf-8 char boundary.
            self.cumulative_output
                .push_str(&string[string.floor_char_boundary(prefix_len)..]);
        }
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
        self.prefix_seeded = true;
        if let Some(truncate_output_to) = truncate_output_to {
            self.cumulative_output.truncate(truncate_output_to);
        }
        let last_chunk = (self.output_index < self.cumulative_output.len())
            .then(|| self.cumulative_output[self.output_index..].to_string());
        self.output_index = 0;
        Ok((last_chunk, take(&mut self.cumulative_output)))
    }

    fn output(&self) -> &str {
        &self.cumulative_output
    }
}
