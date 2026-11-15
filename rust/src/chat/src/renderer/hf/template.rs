// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Chat template support for tokenizers using Jinja2 templates.
//!
//! This module is inlined from SMG's tokenizer crate with local adaptations:
//! - thinking-related detection/state is removed
//! - special tokens are wired to `vllm_text::backends::hf::HfSpecialTokens`

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{self};
use vllm_text::backend::hf::HfSpecialTokens;

use super::error::TemplateError;
use super::format::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption, detect_chat_template_content_format,
};
use super::tojson::hf_tojson_filter;
use crate::renderer::hf::{TemplateMessage, TemplateTool};

type Result<T> = std::result::Result<T, TemplateError>;

/// Build a pre-configured environment with the given template string.
fn build_environment(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();

    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    env.add_template_owned("chat".to_owned(), template)?;

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", hf_tojson_filter);

    Ok(env)
}

#[serde_with::skip_serializing_none]
#[derive(Default, Serialize)]
pub(super) struct TemplateContext<'a> {
    pub(super) messages: &'a [TemplateMessage],
    pub(super) add_generation_prompt: bool,
    pub(super) continue_final_message: bool,
    pub(super) tools: Option<&'a [TemplateTool]>,
    pub(super) documents: Option<&'a [serde_json::Value]>,
    #[serde(flatten)]
    pub(super) special_tokens: Option<&'a HfSpecialTokens>,
    #[serde(flatten)]
    pub(super) template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
}

/// Load chat template from a file (`.jinja` or `.json` containing Jinja).
pub fn load_chat_template(template_path: &Path) -> Result<Option<String>> {
    let content = fs::read_to_string(template_path).map_err(TemplateError::ReadTemplateFile)?;

    if template_path.extension().is_some_and(|ext| ext == "json") {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ChatTemplateFile {
            String(String),
            Object { chat_template: String },
        }

        let json_value =
            serde_json::from_str(&content).map_err(TemplateError::ParseTemplateJson)?;
        let json_template =
            serde_json::from_value(json_value).map_err(|_| TemplateError::InvalidTemplateJson)?;

        return Ok(Some(match json_template {
            ChatTemplateFile::String(template) => template,
            ChatTemplateFile::Object { chat_template } => chat_template,
        }));
    }

    let template = content.trim().replace("\\n", "\n");
    Ok(Some(template))
}

/// Resolve a configured chat template value into a template string.
pub fn resolve_chat_template(chat_template: &str) -> Result<String> {
    let path = Path::new(chat_template);
    if path.exists() {
        return load_chat_template(path).map(|template| template.unwrap_or_default());
    }

    const JINJA_CHARS: [char; 3] = ['{', '}', '\n'];
    if chat_template.chars().any(|c| JINJA_CHARS.contains(&c)) {
        return Ok(chat_template.to_string());
    }

    Err(TemplateError::MissingTemplatePath)
}

/// One compiled chat template with its Jinja environment and detected content
/// format.
pub(super) struct CompiledChatTemplate {
    /// Cached, fully-configured environment for one compiled template.
    env: Environment<'static>,
    content_format: ChatTemplateContentFormat,
}

impl CompiledChatTemplate {
    /// Compile the given chat template string into a [`CompiledChatTemplate`].
    pub fn new(template: String, content_format: ChatTemplateContentFormatOption) -> Result<Self> {
        let content_format = match content_format {
            ChatTemplateContentFormatOption::Auto => detect_chat_template_content_format(&template),
            ChatTemplateContentFormatOption::String => ChatTemplateContentFormat::String,
            ChatTemplateContentFormatOption::OpenAi => ChatTemplateContentFormat::OpenAi,
        };
        let env = build_environment(template)?;
        Ok(Self {
            env,
            content_format,
        })
    }

    /// Apply the compiled template to the given context and return the rendered
    /// prompt.
    pub fn apply(&self, ctx: TemplateContext<'_>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(ctx).map_err(TemplateError::from)
    }

    pub fn content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }
}
