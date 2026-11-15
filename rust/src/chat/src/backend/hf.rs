// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use tracing::info;
use vllm_text::backend::hf::{HfTextBackend, ResolvedModelFiles, load_model_config};
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{DynTextBackend, TextBackend as _};

use crate::backend::{
    ChatBackend, DynChatBackend, LoadModelBackendsOptions, LoadedModelBackends,
    NewChatOutputProcessorOptions,
};
use crate::error::Result;
use crate::multimodal::{MultimodalConfigFiles, MultimodalModelInfo};
use crate::output::{
    DefaultChatOutputProcessor, HarmonyChatOutputProcessor, validate_harmony_parser_overrides,
};
use crate::renderer::hf::{HfChatRenderer, MultimodalRenderInfo};
use crate::renderer::{
    DeepSeekV4ChatRenderer, DeepSeekV32ChatRenderer, DynChatRenderer, HarmonyChatRenderer,
    InklingChatRenderer, KimiK3ChatRenderer,
};
use crate::request::ChatRequest;
use crate::{DynChatOutputProcessor, RendererSelection};

/// [`ChatBackend`] implementation built on Hugging Face model files.
pub struct HfChatBackend {
    model_id: String,
    model_type: String,
    tokenizer: DynTokenizer,
    chat_renderer: DynChatRenderer,
    multimodal_model_info: Option<MultimodalModelInfo>,
}

impl HfChatBackend {
    /// Load the chat backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        options: LoadModelBackendsOptions,
        tokenizer: DynTokenizer,
    ) -> Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let model_type = model_config.model_type().unwrap_or_default();
        let multimodal_model_info = if options.language_model_only {
            None
        } else {
            MultimodalModelInfo::from_paths(
                model_id.clone(),
                (!model_type.is_empty()).then_some(model_type.to_string()),
                MultimodalConfigFiles {
                    config: files.config_path.as_deref(),
                    preprocessor_config: files.preprocessor_config_path.as_deref(),
                    video_preprocessor_config: files.video_preprocessor_config_path.as_deref(),
                    processor_config: files.processor_config_path.as_deref(),
                },
                tokenizer.clone(),
                options.limit_mm_per_prompt.clone(),
            )?
        };
        let multimodal_render_info = resolve_multimodal_render_info(multimodal_model_info.as_ref());

        let renderer = options.renderer.resolve(model_type);
        let chat_renderer: DynChatRenderer = match renderer {
            RendererSelection::Auto => unreachable!("renderer auto should be resolved above"),
            RendererSelection::Hf => Arc::new(HfChatRenderer::load(
                &files,
                options,
                multimodal_render_info,
            )?),
            RendererSelection::DeepSeekV32 => Arc::new(DeepSeekV32ChatRenderer::new()),
            RendererSelection::DeepSeekV4 => Arc::new(DeepSeekV4ChatRenderer::new()),
            RendererSelection::Harmony => Arc::new(HarmonyChatRenderer::new()?),
            RendererSelection::Inkling => Arc::new(InklingChatRenderer::new(tokenizer.clone())?),
            RendererSelection::KimiK3 => Arc::new(KimiK3ChatRenderer::new(tokenizer.clone())),
        };

        info!(
            model_id,
            model_type,
            %renderer,
            "loaded chat backend with Hugging Face model files"
        );

        Ok(Self {
            model_id,
            model_type: model_type.to_string(),
            tokenizer,
            chat_renderer,
            multimodal_model_info,
        })
    }
}

impl ChatBackend for HfChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        self.chat_renderer.clone()
    }

    fn multimodal_model_info(&self) -> Option<&MultimodalModelInfo> {
        self.multimodal_model_info.as_ref()
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> Result<DynChatOutputProcessor> {
        if self.model_type == "gpt_oss" {
            validate_harmony_parser_overrides(options.tool_call_parser, options.reasoning_parser)?;
            return Ok(Box::new(HarmonyChatOutputProcessor::new(request)?));
        }

        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            &self.model_id,
            self.tokenizer.clone(),
            options.tool_call_parser,
            options.reasoning_parser,
        )?))
    }
}

/// Load the Hugging Face text and chat backends for the given model id.
pub(super) async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    let files = ResolvedModelFiles::new(model_id).await?;
    let text_backend = HfTextBackend::from_resolved_model_files(
        files.clone(),
        model_id.to_string(),
        options.generation_config,
    )?;
    let tokenizer = text_backend.tokenizer();
    let text_backend: DynTextBackend = Arc::new(text_backend);

    let chat_backend: DynChatBackend = Arc::new(HfChatBackend::from_resolved_model_files(
        files,
        model_id.to_string(),
        options,
        tokenizer,
    )?);

    Ok(LoadedModelBackends {
        text_backend,
        chat_backend,
    })
}

fn resolve_multimodal_render_info(
    info: Option<&MultimodalModelInfo>,
) -> Option<MultimodalRenderInfo> {
    use llm_multimodal::Modality;

    info.map(|info| MultimodalRenderInfo {
        image_token: info.placeholder_token(Modality::Image).map(str::to_string),
        video_token: info.placeholder_token(Modality::Video).map(str::to_string),
        audio_token: info.placeholder_token(Modality::Audio).map(str::to_string),
    })
}
