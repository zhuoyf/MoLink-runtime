// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod config;
mod model_files;

use std::collections::BTreeSet;
use std::sync::Arc;

use tracing::info;
use vllm_tokenizer::{
    DynTokenizer, HuggingFaceTokenizer, TekkenTokenizer, TiktokenTokenizer, Tokenizer,
};

use self::config::{GenerationConfig, load_generation_config};
pub use self::config::{
    HfSpecialTokens, HfTokenizerConfig, ModelConfig, NamedSpecialToken, load_model_config,
    load_tokenizer_config,
};
pub use self::model_files::{ResolvedModelFiles, TokenizerSource};
use crate::backend::{GenerationConfigMode, SamplingHints, TextBackend};
use crate::error::Result;

fn load_tokenizer(tokenizer: &TokenizerSource) -> Result<DynTokenizer> {
    match tokenizer {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}

/// [`TextBackend`] implementation built on Hugging Face model files.
pub struct HfTextBackend {
    model_id: String,
    files: ResolvedModelFiles,
    tokenizer: DynTokenizer,
    /// Primary EOS handled by engine-core's dedicated EOS path.
    primary_eos_token_id: Option<u32>,
    /// Additional EOS ids that should flow through stop-token handling.
    extra_eos_token_ids: BTreeSet<u32>,
    /// Generation-config for sampling defaults that may be inherited when the
    /// user does not explicitly override them.
    generation_config: GenerationConfig,
    generation_config_mode: GenerationConfigMode,
    /// Model vocabulary size from the selected text config.
    model_vocab_size: usize,
    /// Model config (`config.json`).
    model_config: ModelConfig,
}

impl HfTextBackend {
    /// Load the text backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        generation_config_mode: GenerationConfigMode,
    ) -> Result<Self> {
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let model_config = load_model_config(files.config_path.as_deref())?;
        let model_vocab_size = model_config.vocab_size()? as usize;
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let (primary_eos_token_id, extra_eos_token_ids) = resolve_eos_token_ids(
            &tokenizer_config,
            &model_config,
            &generation_config,
            tokenizer.as_ref(),
        );

        info!(
            model_id,
            "loaded text backend with Hugging Face model files"
        );

        Ok(Self {
            model_id,
            files,
            tokenizer,
            primary_eos_token_id,
            extra_eos_token_ids,
            generation_config,
            generation_config_mode,
            model_vocab_size,
            model_config,
        })
    }

    /// Expose the resolved model files for use by the chat backend to load the
    /// chat template.
    pub fn resolved_model_files(&self) -> &ResolvedModelFiles {
        &self.files
    }
}

/// Resolve EOS hints from tokenizer, model, and generation configs.
///
/// Resolution rules:
/// 1. Use the tokenizer-side EOS token as the primary EOS when it resolves to a
///    token id.
/// 2. Fall back to the first model-config EOS id when the tokenizer config does
///    not provide a primary EOS.
/// 3. Keep any remaining model/generation EOS ids as extra stop-token ids so
///    they still participate in stopping and min-token handling.
fn resolve_eos_token_ids(
    tokenizer_config: &HfTokenizerConfig,
    model_config: &ModelConfig,
    generation_config: &GenerationConfig,
    tokenizer: &dyn Tokenizer,
) -> (Option<u32>, BTreeSet<u32>) {
    let model_config_eos_token_ids = model_config.eos_token_ids();
    let primary_eos_token_id = tokenizer_config
        .special_tokens
        .eos_token
        .as_ref()
        .and_then(|token| tokenizer.token_to_id(token.as_str()))
        .or_else(|| model_config_eos_token_ids.first().copied());

    let mut extra_eos_token_ids = generation_config
        .eos_token_id
        .clone()
        .map(|value| value.into_set())
        .unwrap_or_default();
    extra_eos_token_ids.extend(model_config_eos_token_ids.iter().copied());
    if let Some(primary_eos_token_id) = primary_eos_token_id {
        extra_eos_token_ids.remove(&primary_eos_token_id);
    }

    (primary_eos_token_id, extra_eos_token_ids)
}

impl TextBackend for HfTextBackend {
    fn tokenizer(&self) -> DynTokenizer {
        self.tokenizer.clone()
    }

    fn is_moe(&self) -> bool {
        self.model_config.is_moe()
    }

    fn model_vocab_size(&self) -> usize {
        self.model_vocab_size
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(build_sampling_hints(
            self.generation_config_mode,
            &self.generation_config,
            self.primary_eos_token_id,
            self.extra_eos_token_ids.clone(),
        ))
    }
}

fn build_sampling_hints(
    mode: GenerationConfigMode,
    generation_config: &GenerationConfig,
    primary_eos_token_id: Option<u32>,
    extra_eos_token_ids: BTreeSet<u32>,
) -> SamplingHints {
    let sampling_config = match mode {
        // Auto inherits sampling defaults from the model's generation config.
        GenerationConfigMode::Auto => Some(generation_config),
        // Vllm skips model-provided sampling defaults. `lower_sampling_params`
        // applies vLLM fallback values; separately resolved EOS tokens remain.
        GenerationConfigMode::Vllm => None,
    };

    SamplingHints {
        primary_eos_token_id,
        extra_eos_token_ids,
        default_temperature: sampling_config.and_then(|config| config.temperature),
        default_top_p: sampling_config.and_then(|config| config.top_p),
        default_top_k: sampling_config.and_then(|config| config.top_k),
        default_min_p: sampling_config.and_then(|config| config.min_p),
        default_repetition_penalty: sampling_config.and_then(|config| config.repetition_penalty),
        default_max_tokens: sampling_config.and_then(|config| config.max_new_tokens),
    }
}
