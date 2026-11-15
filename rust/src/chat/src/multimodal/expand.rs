// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prompt placeholder expansion shared across modalities.

use std::collections::{HashMap, VecDeque};

use llm_multimodal::{Modality, PromptReplacement};
use vllm_engine_core_client::protocol::multimodal::PlaceholderRange;
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::PreparedMedia;
use crate::error::{Error, Result, bail_multimodal};

/// One modality's queue of pending placeholder replacements for prompt
/// expansion.
struct ExpansionLane<'a> {
    modality: Modality,
    marker_token_id: u32,
    embed_token_id: u32,
    placeholder_token: String,
    replacements: VecDeque<&'a PromptReplacement>,
}

impl<'a> ExpansionLane<'a> {
    fn from_prepared(media: &'a PreparedMedia) -> Option<Self> {
        if media.replacements.is_empty() {
            return None;
        }

        Some(Self {
            modality: media.modality,
            marker_token_id: media.placeholder.marker_token_id,
            embed_token_id: media.placeholder.embed_token_id,
            placeholder_token: media.placeholder.token.clone(),
            replacements: media.replacements.iter().collect(),
        })
    }
}

/// Replace rendered placeholder markers with model-specific replacement
/// tokens across all modalities in one left-to-right pass.
///
/// Each prepared modality consumes its own marker occurrences in order,
/// matching the original media-part order within that modality; markers of
/// different modalities may interleave freely.
///
/// The returned ranges point into the already-expanded prompt, grouped per
/// modality in item order.
pub(super) fn expand_prompt_token_ids(
    prompt_token_ids: &mut Vec<u32>,
    prepared: &[PreparedMedia],
) -> Result<HashMap<Modality, Vec<PlaceholderRange>>> {
    let mut lanes = prepared.iter().filter_map(ExpansionLane::from_prepared).collect::<Vec<_>>();
    if lanes.is_empty() {
        return Ok(HashMap::new());
    }

    let replacement_growth = lanes
        .iter()
        .flat_map(|lane| lane.replacements.iter())
        .fold(0usize, |total, replacement| {
            total.saturating_add(replacement.tokens.len().saturating_sub(1))
        });
    let expanded_len = prompt_token_ids.len().saturating_add(replacement_growth);

    let mut expanded = Vec::with_capacity(expanded_len);
    let mut ranges = HashMap::<Modality, Vec<PlaceholderRange>>::new();

    for &token in prompt_token_ids.iter() {
        let lane = lanes
            .iter_mut()
            .find(|lane| lane.marker_token_id == token && !lane.replacements.is_empty());
        let Some(lane) = lane else {
            expanded.push(token);
            continue;
        };

        let replacement = lane.replacements.pop_front().expect("lane queue is non-empty");
        debug_assert_eq!(replacement.modality, lane.modality);
        if replacement.tokens.is_empty() {
            bail_multimodal!(
                "placeholder token `{}` expanded to no tokens",
                lane.placeholder_token
            );
        }

        let replacement_len = replacement.tokens.len();
        let structural_prefix = replacement.structural_prefix;
        if structural_prefix > expanded.len() {
            bail_multimodal!(
                "placeholder token `{}` declares a structural prefix of {} token(s), but only {} token(s) precede it",
                lane.placeholder_token,
                structural_prefix,
                expanded.len()
            );
        }
        let range_len = replacement_len.saturating_add(structural_prefix);
        let is_embed = {
            let mut mask = Vec::with_capacity(range_len);
            mask.extend(std::iter::repeat_n(false, structural_prefix));
            mask.extend(
                replacement.tokens.iter().map(|&token| token as u32 == lane.embed_token_id),
            );
            WireTensor::from_bool(vec![range_len], mask).map_err(Error::Multimodal)?
        };

        let expanded_offset = expanded.len() - structural_prefix;
        expanded.extend(replacement.tokens.iter().map(|&token| token as u32));
        ranges.entry(lane.modality).or_default().push(PlaceholderRange {
            offset: expanded_offset,
            length: range_len,
            is_embed: Some(is_embed),
        });
    }

    for lane in &lanes {
        if !lane.replacements.is_empty() {
            bail_multimodal!(
                "placeholder token `{}` was not found in tokenized prompt for {} remaining `{}` item(s)",
                lane.placeholder_token,
                lane.replacements.len(),
                lane.modality
            );
        }
    }

    *prompt_token_ids = expanded;

    Ok(ranges)
}
