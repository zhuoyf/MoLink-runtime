// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use base64::Engine as _;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;
use thiserror_ext::AsReport as _;
use tracing::{info, warn};

use crate::{Result, Tokenizer};

/// Default regex pattern used when loading tiktoken from a BPE file. This is
/// the same `cl100k_base` pattern that HuggingFace transformers uses as its
/// default in `TikTokenConverter`.
const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Kimi BPE pattern from `moonshotai/Kimi-K2-Instruct/tokenization_kimi.py`.
const KIMI_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Fallback number of reserved special-token slots to assume when the model's
/// `config.json` is not available (so we cannot read `vocab_size` directly).
///
/// 256 is the value used by Kimi K2 / K2.5 (`tokenization_kimi.py`'s
/// `num_reserved_special_tokens`) and by Llama 3, and it appears to be the most
/// common convention among modern tiktoken-based HF tokenizers. When
/// `config.json` *is* present we honour the model's actual `vocab_size` instead
/// of this fallback — see `Self::new`.
const FALLBACK_NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;
const DISABLE_RIPTOKEN_ENV: &str = "VLLM_RS_DISABLE_RIPTOKEN";

/// Parsed entry from `tokenizer_config.json`'s `added_tokens_decoder`.
#[derive(Debug, Clone, Deserialize)]
struct AddedToken {
    content: String,
    /// HuggingFace `added_tokens_decoder` entries can be marked `"special":
    /// true|false`. Special tokens are dropped from output when `decode` is
    /// called with `skip_special_tokens = true`. Defaults to `false` when
    /// the field is omitted, matching HuggingFace's `AddedToken` default —
    /// so only tokens explicitly marked special are stripped during normal
    /// decode (where `skip_special_tokens` itself defaults to true).
    #[serde(default)]
    special: bool,
}

/// Minimal subset of `tokenizer_config.json` needed by the tiktoken loader.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TiktokenTokenizerConfig {
    /// Format:
    /// `{ "added_tokens_decoder": { "163584": { "content": "[BOS]", "special":
    /// true }, ... } }`
    #[serde(default)]
    added_tokens_decoder: FxHashMap<u32, AddedToken>,
}

/// Minimal subset of model `config.json` needed by the tiktoken loader.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TiktokenModelConfig {
    model_type: Option<String>,
    vocab_size: Option<u32>,
    text_config: Option<Box<TiktokenModelConfig>>,
}

impl TiktokenModelConfig {
    /// Read `model_type` from a model `config.json` value, falling back to a
    /// single-level nested `text_config.model_type` for composite (e.g.
    /// multimodal) configs that keep text metadata under a `text_config`
    /// object.
    fn effective_model_type(&self) -> Option<&str> {
        self.model_type
            .as_deref()
            .or_else(|| self.text_config.as_deref()?.effective_model_type())
    }

    /// Read `vocab_size` from a model `config.json` value, falling back to a
    /// single-level nested `text_config.vocab_size` for composite (e.g.
    /// multimodal) configs that keep text metadata under a `text_config`
    /// object — matching the same shape `ModelConfig` parses.
    fn effective_vocab_size(&self) -> Option<u32> {
        self.vocab_size.or_else(|| self.text_config.as_deref()?.effective_vocab_size())
    }
}

/// Tiktoken tokenizer from `tiktoken.model` or `*.tiktoken` BPE files.
pub struct TiktokenTokenizer {
    backend: Backend,
    metadata: TokenMetadata,
}

enum Backend {
    Riptoken(RiptokenBackend),
    TiktokenRs(TiktokenRsBackend),
}

struct RiptokenBackend {
    inner: Box<riptoken::CoreBPE>,
    allowed_special_tokens: Vec<String>,
}

struct TiktokenRsBackend {
    inner: Box<tiktoken_rs::CoreBPE>,
    /// Reverse map for special / added token strings populated from the
    /// reserved range. This lets `token_to_id` answer special-token lookups
    /// directly without round-tripping through `tiktoken-rs`'s encoder,
    /// which can panic for unknown special-looking strings.
    special_token_ids_by_text: FxHashMap<String, u32>,
    /// Set of out-of-vocab token IDs we have already warned about. The
    /// reserved-slot population in the constructor should keep this empty
    /// under normal operation; it only fills up if a model emits ids at or
    /// above `vocab_upper_bound` (e.g. an engine sampling bug). We dedupe
    /// so streaming decode (which calls `decode` repeatedly on the same prefix)
    /// does not spam.
    warned_unknown_ids: Mutex<FxHashSet<u32>>,
}

struct TokenMetadata {
    /// Number of regular BPE tokens. Token ids in `[0, num_base_tokens)` are
    /// BPE tokens that always decode to text; ids in `[num_base_tokens,
    /// vocab_upper_bound)` live in the special-token slots and are subject
    /// to `skip_special_tokens` filtering.
    num_base_tokens: u32,
    /// Exclusive upper bound on token IDs that `inner` is guaranteed to know
    /// how to decode.
    ///
    /// The constructor registers every id in `[num_base_tokens,
    /// vocab_upper_bound)` with the inner `CoreBPE` as a (named or
    /// `<|reserved_token_{id}|>`) special token, and the BPE
    /// encoder densely covers `[0, num_base_tokens)`. So any id below this
    /// bound is in one of the inner `CoreBPE`'s decoder maps and
    /// `_decode_native_and_split` will not panic on it. `decode` filters
    /// out ids at or above this bound to keep that guarantee.
    vocab_upper_bound: u32,
    /// Ids in `[num_base_tokens, vocab_upper_bound)` whose
    /// `added_tokens_decoder` entry was explicitly marked `"special":
    /// false` — i.e. tokens that should still appear in output
    /// even when `skip_special_tokens = true`. For Kimi K2 / K2.5 this
    /// typically holds the tool-call markers and `<think>` / `</think>`.
    /// Reserved-slot placeholders are not in this set (they default to
    /// special and get skipped).
    non_special_added_ids: FxHashSet<u32>,
    /// Raw token string by token id. Base BPE tokens are represented with
    /// lossy UTF-8, matching decode behavior for byte sequences that are not
    /// valid UTF-8 on their own.
    token_by_id: FxHashMap<u32, String>,
}

impl TokenMetadata {
    fn filter_special_tokens(&self, token_ids: &[u32]) -> Vec<u32> {
        token_ids
            .iter()
            .copied()
            .filter(|&id| {
                id < self.num_base_tokens
                    || id >= self.vocab_upper_bound
                    || self.non_special_added_ids.contains(&id)
            })
            .collect()
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        token_id >= self.num_base_tokens
            && token_id < self.vocab_upper_bound
            && !self.non_special_added_ids.contains(&token_id)
    }

    fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.token_by_id.get(&token_id).cloned()
    }
}

impl RiptokenBackend {
    fn encode(&self, text: &str) -> Vec<u32> {
        // TODO: avoid collecting `allowed_special` every time this method is called.
        let allowed_special: HashSet<&str> =
            self.allowed_special_tokens.iter().map(String::as_str).collect();
        self.inner.encode(text, &allowed_special)
    }

    fn decode(&self, token_ids: &[u32]) -> String {
        let bytes = self.inner.decode_bytes(token_ids);
        // TODO: use `from_utf8_lossy_owned` once it's stabilized.
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.encode_single_token(token.as_bytes())
    }
}

impl TiktokenRsBackend {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode_with_special_tokens(text)
    }

    fn decode(&self, token_ids: &[u32], metadata: &TokenMetadata) -> String {
        let safe_ids: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&id| {
                if id >= metadata.vocab_upper_bound {
                    self.warn_unknown_id(id);
                    return false;
                }
                true
            })
            .collect();
        let bytes: Vec<u8> = self.inner._decode_native_and_split(safe_ids).flatten().collect();
        // TODO: use `from_utf8_lossy_owned` once it's stabilized.
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(&token_id) = self.special_token_ids_by_text.get(token) {
            return Some(token_id);
        }

        // Fall back to ordinary encoding for regular vocabulary items. This
        // deliberately avoids `encode_with_special_tokens`: older `tiktoken-rs`
        // versions can panic if the input text merely *looks* like a special
        // token but is not registered in `special_tokens_encoder`.
        let ids = self.inner.encode_ordinary(token);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    }

    /// Log a warning the first time an unknown token id is seen during decode,
    /// deduped across calls so streaming decode does not spam the log for
    /// the same id.
    fn warn_unknown_id(&self, token_id: u32) {
        let newly_inserted = self
            .warned_unknown_ids
            .lock()
            .map(|mut set| set.insert(token_id))
            .unwrap_or(false);
        if newly_inserted {
            warn!(
                token_id,
                "tiktoken-rs decode encountered token id not in the vocabulary; skipping. \
                 This typically indicates a sparse-vocab model whose `added_tokens_decoder` \
                 does not list every reserved id in the special-token range."
            );
        }
    }
}

impl TiktokenTokenizer {
    /// Load a tiktoken tokenizer from a `.tiktoken` / `tiktoken.model` BPE
    /// file.
    ///
    /// The BPE file format is one `<base64-token-bytes> <rank>` pair per line,
    /// the same format used by OpenAI's tiktoken and by HuggingFace model
    /// repos that ship tiktoken files (e.g. DeepSeek, Kimi K2).
    ///
    /// Special / added tokens are read from `tokenizer_config.json` in the same
    /// directory when present. The `cl100k_base` regex pattern is used as a
    /// reasonable default.
    pub fn new(path: &Path) -> Result<Self> {
        if std::env::var_os(DISABLE_RIPTOKEN_ENV).is_some() {
            return Self::new_tiktoken_rs(path);
        }

        match Self::new_riptoken(path) {
            Ok(tokenizer) => Ok(tokenizer),
            Err(error) => {
                warn!(
                    path = %path.display(),
                    error = %error.as_report(),
                    "failed to load tokenizer with riptoken; falling back to tiktoken-rs"
                );
                Self::new_tiktoken_rs(path)
            }
        }
    }

    /// Load from `tiktoken.model` / `*.tiktoken` with riptoken.
    pub fn new_riptoken(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with riptoken (BPE file)");

        let config = LoadedTiktokenConfig::load(path)?;
        let allowed_special_tokens = config.special_tokens_encoder.keys().cloned().collect();
        let inner = riptoken::CoreBPE::new(
            config.encoder.into_iter().collect(),
            config.special_tokens_encoder.into_iter().collect(),
            config.pattern,
        )
        .map_err(|error| {
            tokenizer_error!(
                "failed to create riptoken tokenizer from {}: {error}",
                path.display()
            )
        })?;

        Ok(Self {
            backend: Backend::Riptoken(RiptokenBackend {
                inner: Box::new(inner),
                allowed_special_tokens,
            }),
            metadata: config.metadata,
        })
    }

    /// Load from `tiktoken.model` / `*.tiktoken` with tiktoken-rs.
    pub fn new_tiktoken_rs(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with tiktoken-rs (BPE file)");

        let config = LoadedTiktokenConfig::load(path)?;
        let special_token_ids_by_text = config.special_tokens_encoder.clone();
        let inner = tiktoken_rs::CoreBPE::new(
            config.encoder,
            config.special_tokens_encoder,
            config.pattern,
        )
        .map_err(|error| {
            tokenizer_error!(
                "failed to create tiktoken-rs tokenizer from {}: {error}",
                path.display()
            )
        })?;

        Ok(Self {
            backend: Backend::TiktokenRs(TiktokenRsBackend {
                inner: Box::new(inner),
                special_token_ids_by_text,
                warned_unknown_ids: Mutex::new(FxHashSet::default()),
            }),
            metadata: config.metadata,
        })
    }
}

struct LoadedTiktokenConfig {
    encoder: FxHashMap<Vec<u8>, u32>,
    special_tokens_encoder: FxHashMap<String, u32>,
    metadata: TokenMetadata,
    pattern: &'static str,
}

impl LoadedTiktokenConfig {
    fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|error| {
            tokenizer_error!(
                "failed to read tiktoken file {}: {}",
                path.display(),
                error.as_report()
            )
        })?;
        let mut encoder: FxHashMap<Vec<u8>, u32> =
            FxHashMap::with_capacity_and_hasher(content.lines().count(), Default::default());
        for line in content.lines() {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let token_b64 =
                parts.next().ok_or_else(|| tokenizer_error!("missing token in tiktoken file"))?;
            let rank_str =
                parts.next().ok_or_else(|| tokenizer_error!("missing rank in tiktoken file"))?;
            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .map_err(|error| tokenizer_error!("invalid base64 in tiktoken file: {error}"))?;
            let rank: u32 = rank_str
                .parse()
                .map_err(|error| tokenizer_error!("invalid rank in tiktoken file: {error}"))?;
            encoder.insert(token_bytes, rank);
        }

        let parent_dir = path.parent();

        // Read added/special tokens (id -> {name, special}) from
        // tokenizer_config.json in the same dir.
        let added_tokens_by_id = parent_dir
            .map(|dir| dir.join("tokenizer_config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let content = std::fs::read_to_string(&config_path).ok()?;
                serde_json::from_str(&content).ok()
            })
            .map(|config: TiktokenTokenizerConfig| config.added_tokens_decoder)
            .unwrap_or_default();

        let model_config: Option<TiktokenModelConfig> = parent_dir
            .map(|dir| dir.join("config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let content = std::fs::read_to_string(&config_path).ok()?;
                serde_json::from_str(&content).ok()
            });
        let vocab_size_from_config = model_config.as_ref().and_then(|c| c.effective_vocab_size());

        // Build the full special-tokens encoder by populating the reserved
        // range that follows the BPE vocabulary. Unknown reserved slots get
        // Python-compatible placeholder names so sampled ids can still decode.
        //
        // Note: `*.tiktoken` ranks are token ids, and they are not guaranteed
        // to be contiguous. The base-vocab boundary is therefore `max_rank + 1`,
        // not `encoder.len()`.
        let num_base_tokens =
            encoder.values().copied().max().map_or(0, |max_rank| max_rank.saturating_add(1));
        let max_added_id = added_tokens_by_id.keys().copied().max().unwrap_or(0);
        let reserved_end = vocab_size_from_config
            .unwrap_or_else(|| num_base_tokens.saturating_add(FALLBACK_NUM_RESERVED_SPECIAL_TOKENS))
            .max(num_base_tokens)
            .max(max_added_id.saturating_add(1));

        let mut special_tokens_encoder: FxHashMap<String, u32> =
            FxHashMap::with_capacity_and_hasher(
                (reserved_end - num_base_tokens) as usize,
                Default::default(),
            );
        let mut non_special_added_ids: FxHashSet<u32> = FxHashSet::default();
        for id in num_base_tokens..reserved_end {
            let name = match added_tokens_by_id.get(&id) {
                Some(token) => {
                    if !token.special {
                        non_special_added_ids.insert(id);
                    }
                    token.content.clone()
                }
                None => format!("<|reserved_token_{id}|>"),
            };
            special_tokens_encoder.insert(name, id);
        }

        let mut token_by_id: FxHashMap<u32, String> = FxHashMap::with_capacity_and_hasher(
            encoder.len() + special_tokens_encoder.len(),
            Default::default(),
        );
        for (token_bytes, &id) in &encoder {
            token_by_id.insert(id, String::from_utf8_lossy(token_bytes).into_owned());
        }
        for (token, &id) in &special_tokens_encoder {
            token_by_id.insert(id, token.clone());
        }

        let pattern = model_config.as_ref().map_or(CL100K_BASE_PATTERN, detect_bpe_pattern);

        Ok(Self {
            encoder,
            special_tokens_encoder,
            metadata: TokenMetadata {
                num_base_tokens,
                vocab_upper_bound: reserved_end,
                non_special_added_ids,
                token_by_id,
            },
            pattern,
        })
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // Tiktoken does not have a separate add_special_tokens toggle; both
        // backends recognize registered special tokens in the input.
        Ok(match &self.backend {
            Backend::Riptoken(backend) => backend.encode(text),
            Backend::TiktokenRs(backend) => backend.encode(text),
        })
    }

    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        Ok(match &self.backend {
            Backend::Riptoken(backend) => backend.inner.encode_ordinary(text),
            Backend::TiktokenRs(backend) => backend.inner.encode_ordinary(text),
        })
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        // Filter passes:
        //
        // 1. The constructor registers every id in `[num_base_tokens, vocab_upper_bound)` as a
        //    special token (named or `<|reserved_token_{id}|>` placeholder, matching
        //    `tokenization_kimi.py`). The tiktoken-rs backend additionally drops ids at or above
        //    that bound so `_decode_native_and_split` cannot panic; riptoken's `decode_bytes`
        //    already skips unknown ids.
        //
        // 2. When `skip_special_tokens = true`, ids in `[num_base_tokens, vocab_upper_bound)` are
        //    dropped *unless* they were marked `"special": false` in `added_tokens_decoder`. This
        //    matches HuggingFace's tokenizer semantics: tool-call markers and `<think>` /
        //    `</think>` (which Kimi K2 / K2.5 declare as non-special) stay in the output, while
        //    BOS/EOS/header tokens and reserved-slot placeholders are stripped.
        //
        // Lossy UTF-8 decoding (instead of strict `String::from_utf8`) is used so
        // partial multi-byte sequences become `\u{FFFD}`, which `DecodeStream`
        // relies on to detect incomplete characters during streaming.
        let ids = if skip_special_tokens {
            &self.metadata.filter_special_tokens(token_ids)
        } else {
            token_ids
        };

        Ok(match &self.backend {
            Backend::Riptoken(backend) => backend.decode(ids),
            Backend::TiktokenRs(backend) => backend.decode(ids, &self.metadata),
        })
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.backend {
            Backend::Riptoken(backend) => backend.token_to_id(token),
            Backend::TiktokenRs(backend) => backend.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.metadata.id_to_token(id)
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.metadata.is_special_id(token_id)
    }

    fn vocab_size(&self) -> usize {
        // Exclusive upper bound on token ids the tokenizer can decode (BPE base
        // tokens plus the registered special/reserved slots), used to range-check
        // `allowed_token_ids` so tiktoken models are not exempt from validation.
        self.metadata.vocab_upper_bound as usize
    }
}

/// Select the BPE regex pattern for a tiktoken model based on `config.json`.
///
/// Most tiktoken models use the `cl100k_base` regex. Kimi models ship a custom
/// regex in their Python tokenizer implementation; we mirror the explicit
/// `model_type` switch used by Dynamo instead of heuristically parsing Python
/// source files.
fn detect_bpe_pattern(config: &TiktokenModelConfig) -> &'static str {
    let model_type = config.effective_model_type();

    match model_type {
        Some("kimi" | "kimi_k2" | "kimi_k25" | "kimi_k3" | "deepseek_v3") => KIMI_PATTERN,
        _ => CL100K_BASE_PATTERN,
    }
}
