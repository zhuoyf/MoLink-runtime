// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde::Serialize;
use serde_json::Value as JsonValue;
use thiserror_ext::AsReport as _;
use tracing::{info, trace, warn};
use vllm_text::Prompt;
use vllm_text::backend::hf::{
    HfSpecialTokens, HfTokenizerConfig, ResolvedModelFiles, load_tokenizer_config,
};

use self::format::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption as ContentFormatOption,
};
use self::template::{CompiledChatTemplate, TemplateContext};
use super::{ChatRenderer, RenderedPrompt, effective_template_kwargs};
use crate::error::Result;
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};
use crate::{
    AssistantContentBlock, AssistantMessageExt, ChatTool, Error, LoadModelBackendsOptions,
};

mod error;
mod format;
mod template;
mod tojson;

pub use template::{load_chat_template, resolve_chat_template};

pub use self::format::ChatTemplateContentFormatOption;

/// Template-visible placeholder tokens per supported modality.
///
/// A `None` token means the loaded model does not support that modality, and
/// content parts of that modality are rejected during rendering.
#[derive(Debug, Clone, Default)]
pub struct MultimodalRenderInfo {
    pub image_token: Option<String>,
    pub video_token: Option<String>,
    pub audio_token: Option<String>,
}

/// Hugging Face chat-template renderer backed by the local Jinja chat-template
/// state.
pub struct HfChatRenderer {
    default_template: Option<CompiledChatTemplate>,
    default_template_kwargs: HashMap<String, JsonValue>,
    content_format: ContentFormatOption,
    special_tokens: Option<HfSpecialTokens>,
    multimodal: Option<MultimodalRenderInfo>,
}

impl HfChatRenderer {
    /// Create a renderer from the given template string.
    pub fn new(
        template: Option<String>,
        default_template_kwargs: HashMap<String, JsonValue>,
        content_format: ContentFormatOption,
    ) -> Result<Self> {
        Ok(Self {
            default_template: template
                .map(|template| {
                    CompiledChatTemplate::new(template, content_format)
                        .map_err(|error| Error::ChatTemplate(error.to_report_string()))
                })
                .transpose()?,
            default_template_kwargs,
            content_format,
            special_tokens: None,
            multimodal: None,
        })
    }

    pub fn with_special_tokens(mut self, special_tokens: Option<HfSpecialTokens>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    pub fn with_multimodal(mut self, multimodal: Option<MultimodalRenderInfo>) -> Self {
        self.multimodal = multimodal;
        self
    }

    /// Create a renderer from the given model files and loading options.
    pub fn load(
        files: &ResolvedModelFiles,
        options: LoadModelBackendsOptions,
        multimodal: Option<MultimodalRenderInfo>,
    ) -> Result<Self> {
        let HfTokenizerConfig {
            special_tokens,
            chat_template,
            ..
        } = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let mut template = chat_template;
        let special_tokens = (!special_tokens.is_empty()).then_some(special_tokens);

        if let Some(configured_template) = options.chat_template.as_deref() {
            template = Some(
                resolve_chat_template(configured_template)
                    .map_err(|error| Error::ChatTemplate(error.to_report_string()))?,
            );
            info!("using configured chat template override");
        } else if let Some(chat_template_path) = files.chat_template_path.as_deref() {
            // If independent chat template file(s) exist and contain non-empty content,
            // they take priority over template entries in the tokenizer config
            let file_template = load_chat_template(chat_template_path)
                .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

            if file_template.as_ref().is_some_and(|t| !t.trim().is_empty()) {
                info!(
                    path = %chat_template_path.display(),
                    "loaded dedicated chat template file, overriding tokenizer_config chat_template"
                );
                template = file_template;
            } else {
                warn!(
                    path = %chat_template_path.display(),
                    "ignoring empty dedicated chat template file and falling back to tokenizer_config chat_template"
                );
            }
        }

        Ok(Self::new(
            template,
            options.default_chat_template_kwargs,
            options.chat_template_content_format,
        )?
        .with_special_tokens(special_tokens)
        .with_multimodal(multimodal))
    }

    /// Apply the chat template to one chat request, rendering the prompt string
    /// to be tokenized and submitted to the model.
    ///
    /// If the request carries a per-request `chat_template` override, a
    /// temporary template is compiled from that string and used instead of
    /// the model's default.
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        let override_template = request
            .chat_options
            .chat_template
            .as_ref()
            .map(|template| {
                CompiledChatTemplate::new(template.clone(), self.content_format)
                    .map_err(|error| Error::ChatTemplate(error.to_report_string()))
            })
            .transpose()?;
        let template = override_template
            .as_ref()
            .or(self.default_template.as_ref())
            .ok_or(Error::MissingChatTemplate)?;

        self.apply_chat_template_inner(template, request)
    }

    fn apply_chat_template_inner(
        &self,
        effective_template: &CompiledChatTemplate,
        request: &ChatRequest,
    ) -> Result<RenderedPrompt> {
        let mut messages = to_template_messages(
            &request.messages,
            effective_template.content_format(),
            self.multimodal.as_ref(),
        )?;

        // Handling of `continue_final_message`:
        // Append a sentinel tag to the final message content, render as usual, then
        // truncate the rendered prompt at the tag so any template suffix after the
        // final message content (e.g. the end-of-turn marker) is dropped.
        let final_message_text = if request.chat_options.continue_final_message() {
            let final_message = messages.last_mut().ok_or(Error::EmptyMessages)?;
            Some(append_continue_final_message_tag(final_message)?)
        } else {
            None
        };

        let tools = request.tool_parsing_enabled().then(|| to_template_tools(request.tools()));
        trace!(
            message_count = messages.len(),
            content_format = ?effective_template.content_format(),
            ?messages,
            ?tools,
            "applying chat template"
        );

        let effective_template_kwargs =
            effective_template_kwargs(&self.default_template_kwargs, request);
        let prompt = effective_template
            .apply(TemplateContext {
                messages: &messages,
                add_generation_prompt: request.chat_options.add_generation_prompt(),
                continue_final_message: request.chat_options.continue_final_message(),
                tools: tools.as_deref(),
                documents: request.documents.as_deref(),
                template_kwargs: Some(&effective_template_kwargs),
                special_tokens: self.special_tokens.as_ref(),
            })
            .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

        let prompt = match &final_message_text {
            Some(final_message_text) => {
                truncate_prompt_at_continue_final_message_tag(prompt, final_message_text)?
            }
            None => prompt,
        };

        trace!(
            prompt_len = prompt.len(),
            prompt, "rendered chat template prompt"
        );

        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
            effective_template_kwargs,
        })
    }
}

impl ChatRenderer for HfChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        self.apply_chat_template(request)
    }
}

/// Chat message in the JSON shape expected by Jinja chat templates.
// TODO: borrow more fields directly from the original `ChatMessage`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct TemplateMessage {
    role: &'static str,
    content: TemplateContent,
    // Developer-role messages may provide message-local tools in the same shape
    // as top-level request tools.
    tools: Option<Vec<TemplateTool>>,
    // Reasoning-capable HF templates are inconsistent on the exact field name,
    // so expose both variants for compatibility.
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    // Function-call-capable templates commonly expect assistant tool calls
    // under this OpenAI-compatible field name.
    tool_calls: Option<Vec<TemplateToolCall>>,
    // Tool-role messages refer back to the assistant call they are answering.
    tool_call_id: Option<String>,
}

/// Chat content in the two shapes HF templates commonly expect.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum TemplateContent {
    String(String),
    OpenAi(Vec<TemplateContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TemplateContentPart {
    Text { text: String },
    Image,
    Video,
    Audio,
}

#[derive(Debug, Serialize)]
struct TemplateToolCall {
    id: String,
    r#type: &'static str, // always "function"
    function: TemplateToolFunction,
}

#[derive(Debug, Serialize)]
struct TemplateToolFunction {
    name: String,
    arguments: JsonValue,
}

#[derive(Debug, Serialize)]
pub(super) struct TemplateTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: TemplateToolDefinition,
}

#[derive(Debug, Serialize)]
struct TemplateToolDefinition {
    name: String,
    description: Option<String>,
    parameters: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// Convert chat messages into the JSON shape expected by Jinja chat templates.
fn to_template_messages(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateMessage>> {
    messages
        .iter()
        .map(|message| to_template_message(message, content_format, multimodal))
        .collect()
}

fn to_template_message(
    message: &ChatMessage,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateMessage> {
    Ok(match message {
        ChatMessage::System { content } => TemplateMessage {
            role: "system",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::Developer { content, tools } => TemplateMessage {
            role: "developer",
            content: to_template_content(content, content_format, multimodal)?,
            tools: tools.as_deref().map(to_template_tools),
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::User { content } => TemplateMessage {
            role: "user",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::Assistant { content } => {
            let text = content.text();
            let reasoning = content.reasoning();
            let tool_calls = to_template_tool_calls(content)?;
            let content =
                to_template_content(&ChatContent::Text(text), content_format, multimodal)?;
            TemplateMessage {
                role: "assistant",
                content,
                tools: None,
                reasoning: reasoning.clone(),
                reasoning_content: reasoning,
                tool_calls,
                tool_call_id: None,
            }
        }
        ChatMessage::ToolResponse {
            content,
            tool_call_id,
        } => TemplateMessage {
            role: "tool",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.clone()),
        },
    })
}

fn to_template_tool_calls(
    content: &[AssistantContentBlock],
) -> Result<Option<Vec<TemplateToolCall>>> {
    let mut tool_calls = Vec::new();

    for tool_call in content.tool_calls() {
        let arguments = serde_json::from_str(&tool_call.arguments).map_err(|error| {
            Error::ChatTemplate(format!(
                "assistant tool call `{}` has invalid JSON arguments: {}",
                tool_call.id,
                error.as_report()
            ))
        })?;
        tool_calls.push(TemplateToolCall {
            id: tool_call.id.clone(),
            r#type: "function",
            function: TemplateToolFunction {
                name: tool_call.name.clone(),
                arguments,
            },
        });
    }

    Ok((!tool_calls.is_empty()).then_some(tool_calls))
}

fn to_template_content(
    content: &ChatContent,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateContent> {
    Ok(match content_format {
        ChatTemplateContentFormat::String => {
            TemplateContent::String(to_template_string_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::OpenAi => {
            TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
        }
    })
}

fn to_template_openai_content(
    content: &ChatContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateContentPart>> {
    match content {
        ChatContent::Text(text) => Ok(vec![TemplateContentPart::Text { text: text.clone() }]),
        ChatContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ChatContentPart::Text { text } => {
                    Ok(TemplateContentPart::Text { text: text.clone() })
                }
                // All multimodal contents are normalized to `{ "type": <modality> }`.
                ChatContentPart::ImageUrl { .. } => {
                    multimodal
                        .and_then(|multimodal| multimodal.image_token.as_ref())
                        .ok_or(Error::UnsupportedMultimodalContent("image_url"))?;
                    Ok(TemplateContentPart::Image)
                }
                ChatContentPart::VideoUrl { .. } => {
                    multimodal
                        .and_then(|multimodal| multimodal.video_token.as_ref())
                        .ok_or(Error::UnsupportedMultimodalContent("video_url"))?;
                    Ok(TemplateContentPart::Video)
                }
                ChatContentPart::InputAudio { .. } | ChatContentPart::AudioUrl { .. } => {
                    multimodal
                        .and_then(|multimodal| multimodal.audio_token.as_ref())
                        .ok_or(Error::UnsupportedMultimodalContent("audio"))?;
                    Ok(TemplateContentPart::Audio)
                }
            })
            .collect(),
    }
}

fn to_template_string_content(
    content: &ChatContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<String> {
    match content {
        ChatContent::Text(text) => Ok(text.clone()),
        ChatContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => out.push_str(text),
                    ChatContentPart::ImageUrl { .. } => {
                        let image_token = multimodal
                            .and_then(|multimodal| multimodal.image_token.as_ref())
                            .ok_or(Error::UnsupportedMultimodalContent("image_url"))?;
                        out.push_str(image_token);
                    }
                    ChatContentPart::VideoUrl { .. } => {
                        let video_token = multimodal
                            .and_then(|multimodal| multimodal.video_token.as_ref())
                            .ok_or(Error::UnsupportedMultimodalContent("video_url"))?;
                        out.push_str(video_token);
                    }
                    ChatContentPart::InputAudio { .. } | ChatContentPart::AudioUrl { .. } => {
                        let audio_token = multimodal
                            .and_then(|multimodal| multimodal.audio_token.as_ref())
                            .ok_or(Error::UnsupportedMultimodalContent("audio"))?;
                        out.push_str(audio_token);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Sentinel appended to the final message content when `continue_final_message`
/// is requested, used to locate the truncation point in the rendered prompt.
///
/// Same literal as `transformers`. Occurrences of this string earlier in the
/// prompt are harmless because truncation uses the rightmost match, and the
/// appended sentinel ends up last as long as the template renders messages in
/// order.
const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Append [`CONTINUE_FINAL_MESSAGE_TAG`] to the trailing text of the final
/// message, returning the original text for post-render validation.
// TODO: transformers v5 also allows continuing a non-`content` field (e.g.
// `reasoning_content`) by passing a field name; only the boolean form is
// supported here.
fn append_continue_final_message_tag(message: &mut TemplateMessage) -> Result<String> {
    let text = match &mut message.content {
        TemplateContent::String(text) => Some(text),
        // Pick the last text part in the message.
        TemplateContent::OpenAi(parts) => parts.iter_mut().rev().find_map(|part| match part {
            TemplateContentPart::Text { text } => Some(text),
            TemplateContentPart::Image
            | TemplateContentPart::Video
            | TemplateContentPart::Audio => None,
        }),
    };
    let text = text.ok_or_else(|| {
        Error::ChatTemplate(
            "continue_final_message is set but there is no text to continue \
             in the final message"
                .to_string(),
        )
    })?;

    let original = text.clone();
    text.push_str(CONTINUE_FINAL_MESSAGE_TAG);
    Ok(original)
}

/// Truncate the rendered prompt at [`CONTINUE_FINAL_MESSAGE_TAG`] so that it
/// ends exactly with the final message content, dropping any template suffix
/// such as end-of-turn markers.
fn truncate_prompt_at_continue_final_message_tag(
    mut rendered: String,
    final_message_text: &str,
) -> Result<String> {
    let tag_loc = rendered
        .rfind(CONTINUE_FINAL_MESSAGE_TAG.trim_end())
        .filter(|_| rendered.contains(final_message_text.trim()));
    let Some(tag_loc) = tag_loc else {
        return Err(Error::ChatTemplate(format!(
            "continue_final_message is set but the final message does not appear \
             in the prompt after applying the chat template! This can happen if \
             the chat template deletes portions of the final message. Final \
             message to continue: {}",
            final_message_text.trim(),
        )));
    };

    if rendered[tag_loc..].starts_with(CONTINUE_FINAL_MESSAGE_TAG) {
        // The template preserved spacing, so a plain cut at the tag suffices.
        rendered.truncate(tag_loc);
    } else {
        // The template trimmed the trailing spacing of the message content, so
        // apply the same trimming to the retained prefix.
        rendered.truncate(tag_loc);
        rendered.truncate(rendered.trim_end().len());
    }
    Ok(rendered)
}

fn to_template_tools(tools: &[ChatTool]) -> Vec<TemplateTool> {
    tools
        .iter()
        .map(|tool| TemplateTool {
            tool_type: "function",
            function: TemplateToolDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            },
        })
        .collect()
}
