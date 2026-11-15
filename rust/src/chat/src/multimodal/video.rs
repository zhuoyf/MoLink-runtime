// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Video-modality preparation: per-clip preprocessing, config resolution,
//! and per-item feature build.

use std::sync::Arc;

use itertools::izip;
use llm_multimodal::{FieldLayout, Modality, PreprocessedEncoderInputs, VideoClip};
use thiserror_ext::AsReport as _;
use tracing::warn;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmFlatField, MmKwargsItem, MmSharedField, MmSlice,
    SliceSpec,
};

use super::{ModalitySupport, MultimodalModelInfo, PreparedItem, PreparedMedia, tensor};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary video encoder input.
///
/// Video-capable vLLM models read `pixel_values_videos` alongside
/// `video_grid_thw`, mirroring the HF processor output naming.
pub(super) const VIDEO_PRIMARY_KEY: &str = "pixel_values_videos";

impl MultimodalModelInfo {
    /// Preprocess fetched video clips one at a time and build per-item
    /// features.
    ///
    /// Unlike images, each clip runs through the preprocessor independently
    /// (a batch of one), so its tensors are complete per item and need no
    /// cross-item slicing.
    pub(super) async fn prepare_videos(
        &self,
        clips: Vec<Arc<VideoClip>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        let support = self.video.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Video.to_string(),
        })?;
        let mut replacements = Vec::with_capacity(clips.len());
        let mut items = Vec::with_capacity(clips.len());

        for (clip, uuid) in izip!(&clips, uuids) {
            let preprocessed = self.preprocess_video_clip(support, Arc::clone(clip)).await?;
            let mut clip_replacements =
                support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
            if clip_replacements.len() != 1 {
                bail_multimodal!(
                    "expected exactly one prompt replacement per video clip, got {}",
                    clip_replacements.len()
                );
            }
            replacements.push(clip_replacements.pop().unwrap());
            items.push(build_video_item(
                support,
                preprocessed,
                clip.hash.clone(),
                uuid,
                model_dtype,
            )?);
        }

        Ok(PreparedMedia {
            modality: Modality::Video,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess one decoded video clip with the model's resolved vision
    /// processor.
    async fn preprocess_video_clip(
        &self,
        support: &ModalitySupport,
        clip: Arc<VideoClip>,
    ) -> Result<PreprocessedEncoderInputs> {
        let config = support.config.clone();
        let processor = support.processor;

        tokio::task::spawn_blocking(move || {
            // Prefer the borrowed-RGB fast path, which avoids materializing a
            // `DynamicImage` per sampled frame after media decode.
            if let Some(rgb_video) = clip.rgb_video() {
                match rgb_video.frame_refs() {
                    Ok(frame_refs) => match processor.preprocess_video_rgb(&frame_refs, &config) {
                        Ok(preprocessed) => return Ok(preprocessed),
                        Err(error) => warn!(
                            error = %error.as_report(),
                            "RGB video preprocessing fast path failed; falling back to materialized frames"
                        ),
                    },
                    Err(error) => warn!(
                        error,
                        "RGB video frame refs are invalid; falling back to materialized frames"
                    ),
                }
            }

            let frames = clip.materialized_frames().map_err(|error| multimodal!("{error}"))?;
            Ok(processor.preprocess_video(&frames, &config)?)
        })
        .await
        .map_err(|error| multimodal!("video preprocessing task failed: {error}"))?
    }
}

/// Convert one preprocessed video clip into engine kwargs.
///
/// The clip is a batch of one, so no per-item slicing is required: the
/// primary tensor ships as a full-range flat field (the engine re-batches
/// flat fields by concatenating along the declared dim, matching vLLM's
/// `flat_from_sizes` treatment of video patches), and batched metadata
/// tensors drop their singleton batch axis.
fn build_video_item(
    support: &ModalitySupport,
    preprocessed: PreprocessedEncoderInputs,
    hash: String,
    uuid: Option<String>,
    model_dtype: ModelDtype,
) -> Result<PreparedItem> {
    let tensors = tensor::collect_tensors(preprocessed, support.spec.primary_key(), model_dtype)?;

    let mut data = MmKwargsItem::new();
    for (key, tensor) in tensors {
        let keep_on_cpu = support.spec.keep_on_cpu_keys.contains(&key);
        let (value, field) = match support.spec.field_layout_for(&key) {
            Some(FieldLayout::Batched) => (
                tensor.batched_wire_value_at(0)?,
                MmField::Batched(MmBatchedField { keep_on_cpu }),
            ),
            Some(FieldLayout::Flat { .. }) => {
                let len = tensor
                    .first_dim()
                    .ok_or_else(|| multimodal!("flat video input `{key}` is not a tensor"))?;
                (
                    (&tensor).try_into()?,
                    MmField::Flat(MmFlatField {
                        slices: vec![MmSlice::Slice(SliceSpec {
                            start: Some(0),
                            stop: Some(len as isize),
                            step: None,
                        })],
                        dim: 0,
                        keep_on_cpu,
                    }),
                )
            }
            None => (
                (&tensor).try_into()?,
                MmField::Shared(MmSharedField {
                    batch_size: 1,
                    keep_on_cpu,
                }),
            ),
        };

        data.insert(
            key,
            MmFieldElem {
                data: Some(value),
                field,
            },
        );
    }

    Ok(PreparedItem { data, hash, uuid })
}
