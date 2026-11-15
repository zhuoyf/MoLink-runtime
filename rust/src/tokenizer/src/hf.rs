// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use fastokens::Tokenizer as FastokensTokenizer;
use fastokens::decoders::Decoder as FastokensDecoder;
use fastokens::pre_tokenized::{
    PreTokenizedString as FastokensPreTokenizedString, Split as FastokensSplit,
};
use fastokens::{PreTokenizer as FastokensPreTokenizer, Split as FastokensSplitPreTokenizer};
use thiserror_ext::AsReport as _;
use tokenizers::{
    AddedVocabulary, Model as _, OffsetType, PreTokenizer as _, Tokenizer as HfTokenizer,
};
use tracing::{info, warn};

use crate::byte_level_decode::decode_byte_level;
use crate::hf::added_tokens::load_tokenizer_json_with_extra_tokens;
use crate::{Result, Tokenizer};

mod added_tokens;

static EMPTY_HF_ADDED_VOCABULARY: LazyLock<AddedVocabulary> = LazyLock::new(AddedVocabulary::new);

enum Backend {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
    /// Fastokens tokenizer whose decoder is pure GPT-2 byte-level, so we can
    /// bypass `Decoder::decode`'s `Vec<String>`/`join("")` assembly.
    FastokensByteLevel(Box<FastokensTokenizer>),
}

/// True if `dec` is effectively a single `ByteLevel` stage — one `ByteLevel`
/// leaf in a tree of `Sequence`s (fastokens represents `Fuse` as an empty
/// `Sequence`, which is a no-op for our purposes).
fn is_byte_level_only(dec: &FastokensDecoder) -> bool {
    fn count_byte_level(dec: &FastokensDecoder) -> usize {
        match dec {
            FastokensDecoder::ByteLevel(_) => 1,
            FastokensDecoder::Sequence(steps) => steps.iter().map(count_byte_level).sum(),
        }
    }
    count_byte_level(dec) == 1
}

fn decode_fastokens_byte_level(
    t: &FastokensTokenizer,
    token_ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String> {
    let tokens: Vec<&str> = token_ids
        .iter()
        .filter(|&&id| !(skip_special_tokens && t.is_special_token(id)))
        .map(|&id| {
            t.id_to_token(id)
                .ok_or_else(|| tokenizer_error!("decoding failed: unknown token ID: {id}"))
        })
        .collect::<Result<_>>()?;
    Ok(decode_byte_level(tokens))
}

fn encode_hf_ordinary(tokenizer: &HfTokenizer, text: &str) -> tokenizers::Result<Vec<u32>> {
    let mut pretokenized =
        EMPTY_HF_ADDED_VOCABULARY.extract_and_normalize(tokenizer.get_normalizer(), text);

    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
        pre_tokenizer.pre_tokenize(&mut pretokenized)?;
    }
    pretokenized.tokenize(|normalized| tokenizer.get_model().tokenize(normalized.get()))?;
    let encoding = pretokenized.into_encoding(None, 0, OffsetType::Byte)?;
    let encoding = tokenizer.post_process(encoding, None, false)?;
    Ok(encoding.get_ids().to_vec())
}

fn fastokens_fused_split(tokenizer: &FastokensTokenizer) -> Option<&FastokensSplitPreTokenizer> {
    // Keep this predicate aligned with fastokens::Tokenizer::detect_fused_byte_level.
    let FastokensPreTokenizer::Sequence(steps) = tokenizer.pre_tokenizer()? else {
        return None;
    };
    let [
        FastokensPreTokenizer::Split(split),
        FastokensPreTokenizer::ByteLevel(byte_level),
    ] = steps.as_slice()
    else {
        return None;
    };
    byte_level.is_bulk_only().then_some(split)
}

fn fastokens_pre_tokenized_ordinary(
    tokenizer: &FastokensTokenizer,
    text: &str,
) -> FastokensPreTokenizedString {
    // This is fastokens::Tokenizer::build_pre_tokenized with added_tokens = None.
    let normalized = tokenizer
        .normalizer()
        .map_or(Cow::Borrowed(text), |normalizer| normalizer.normalize(text));
    match normalized {
        Cow::Borrowed(_) => FastokensPreTokenizedString::from_text(text),
        Cow::Owned(text) => {
            let len = text.len();
            FastokensPreTokenizedString::new(
                text,
                vec![FastokensSplit {
                    range: 0..len,
                    token_id: None,
                }],
            )
        }
    }
}

fn encode_fastokens_ordinary(
    tokenizer: &FastokensTokenizer,
    text: &str,
) -> std::result::Result<Vec<u32>, fastokens::Error> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut pretokenized = fastokens_pre_tokenized_ordinary(tokenizer, text);
    let ids = if let Some(split) = fastokens_fused_split(tokenizer) {
        split.pre_tokenize(&mut pretokenized)?;
        pretokenized
            .tokenize_batched(|buffer, splits, output| {
                tokenizer.model().tokenize_batch_fused(buffer, splits, output)
            })
            .map_err(fastokens::Error::Model)?
    } else {
        if let Some(pre_tokenizer) = tokenizer.pre_tokenizer() {
            pre_tokenizer.pre_tokenize(&mut pretokenized)?;
        }
        pretokenized
            .tokenize(|text, output| tokenizer.model().tokenize_into(text, output))
            .map_err(fastokens::Error::Model)?
    };

    Ok(tokenizer.post_process(ids, false))
}

/// Tokenizer from `tokenizer.json` in HuggingFace format.
///
/// This tries to load with `fastokens` first for better performance, then falls
/// back to HuggingFace's `tokenizers` if the former fails (e.g. due to
/// unsupported tokenizer features or file formats).
pub struct HuggingFaceTokenizer {
    backend: Backend,
    special_token_ids: Arc<[u32]>,
    added_vocab: Box<[(String, u32)]>,
}

impl HuggingFaceTokenizer {
    fn from_hf_backend(tokenizer: HfTokenizer) -> Self {
        let added_vocab = {
            let mut vocab: Vec<_> = tokenizer
                .get_added_tokens_decoder()
                .iter()
                .map(|(&id, token)| (token.content.clone(), id))
                .collect();
            vocab.sort_unstable_by_key(|(_, id)| *id);
            vocab.into_boxed_slice()
        };
        let special_token_ids = {
            let mut ids: Vec<u32> = tokenizer
                .get_added_tokens_decoder()
                .iter()
                .filter(|(_id, token)| token.special)
                .map(|(id, _token)| *id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            Arc::from(ids)
        };
        Self {
            backend: Backend::Hf(Box::new(tokenizer)),
            special_token_ids,
            added_vocab,
        }
    }

    fn from_fastokens_backend(tokenizer: FastokensTokenizer) -> Self {
        let added_vocab = {
            let mut vocab: Vec<_> = tokenizer
                .added_tokens()
                .into_iter()
                .flat_map(|added_tokens| added_tokens.iter())
                .map(|token| (token.content.to_string(), token.id))
                .collect();
            vocab.sort_unstable_by_key(|(_, id)| *id);
            vocab.into_boxed_slice()
        };
        let special_token_ids = {
            let mut ids: Vec<u32> = tokenizer
                .added_tokens()
                .into_iter()
                .flat_map(|added_tokens| added_tokens.iter())
                .filter(|token| token.special)
                .map(|token| token.id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            Arc::from(ids)
        };
        let byte_level = tokenizer.decoder().is_some_and(is_byte_level_only);
        let backend = if byte_level {
            Backend::FastokensByteLevel(Box::new(tokenizer))
        } else {
            Backend::Fastokens(Box::new(tokenizer))
        };
        Self {
            backend,
            special_token_ids,
            added_vocab,
        }
    }

    /// Load from `tokenizer.json` with `fastokens`.
    pub fn new_fastokens(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with fastokens");
        let tokenizer_json = load_tokenizer_json_with_extra_tokens(path)?;
        let t = FastokensTokenizer::from_json(tokenizer_json)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        Ok(Self::from_fastokens_backend(t))
    }

    /// Load from `tokenizer.json` with Hugging Face `tokenizers`.
    pub fn new_hf(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with huggingface tokenizers");
        let tokenizer_json = load_tokenizer_json_with_extra_tokens(path)?;
        let t = serde_json::from_value::<HfTokenizer>(tokenizer_json)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        Ok(Self::from_hf_backend(t))
    }

    /// Load from `tokenizer.json` via fastokens or HuggingFace tokenizers.
    pub fn new(path: &Path) -> Result<Self> {
        match Self::new_fastokens(path) {
            Ok(tokenizer) => Ok(tokenizer),
            Err(error) => {
                warn!(
                    path = %path.display(),
                    error = %error.as_report(),
                    "failed to load tokenizer with fastokens; falling back to HuggingFace tokenizers"
                );
                Self::new_hf(path)
            }
        }
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match &self.backend {
            Backend::Hf(t) => {
                let encoding = t
                    .encode(text, add_special_tokens)
                    .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))?;
                Ok(encoding.get_ids().to_vec())
            }
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => t
                .encode_with_special_tokens(text, add_special_tokens)
                .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report())),
        }
    }

    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        match &self.backend {
            Backend::Hf(tokenizer) => encode_hf_ordinary(tokenizer, text)
                .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report())),
            Backend::Fastokens(tokenizer) | Backend::FastokensByteLevel(tokenizer) => {
                encode_fastokens_ordinary(tokenizer, text)
                    .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))
            }
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match &self.backend {
            Backend::Hf(t) => t
                .decode(token_ids, skip_special_tokens)
                .map_err(|error| tokenizer_error!("decoding failed: {}", error.as_report())),
            Backend::Fastokens(t) => t
                .decode(token_ids, skip_special_tokens)
                .map_err(|error| tokenizer_error!("decoding failed: {}", error.as_report())),
            Backend::FastokensByteLevel(t) => {
                decode_fastokens_byte_level(t, token_ids, skip_special_tokens)
            }
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.backend {
            Backend::Hf(t) => t.token_to_id(token),
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => t.token_to_id(token),
        }
    }

    fn vocab_size(&self) -> usize {
        match &self.backend {
            Backend::Hf(t) => t.get_vocab_size(true),
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => t.vocab_size(),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match &self.backend {
            Backend::Hf(t) => t.id_to_token(id),
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => {
                t.id_to_token(id).map(ToOwned::to_owned)
            }
        }
    }

    fn added_vocab(&self) -> &[(String, u32)] {
        &self.added_vocab
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.special_token_ids.binary_search(&token_id).is_ok()
    }
}
