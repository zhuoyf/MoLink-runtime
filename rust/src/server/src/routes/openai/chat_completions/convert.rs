// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use itertools::Itertools as _;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart,
    ChatMessage as VllmChatMessage, ChatOptions, ChatRequest, ChatTool, ChatToolChoice,
    GenerationPromptMode, ResolvedToolContext, SamplingParams,
};

use super::types::ChatCompletionRequest;
use super::validate;
use crate::error::{ApiError, bail_invalid_request, chat_submit_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::resolve_generation_prompt_truncation;
use crate::routes::openai::utils::structured_outputs::convert_from_response_format;
use crate::routes::openai::utils::types::{
    ChatMessage, ContentPart, MessageContent, Tool, ToolChoice, ToolChoiceValue, ToolReference,
};
use crate::utils::{
    ResolvedRequestContext, convert_logit_bias, merge_ec_transfer_params, merge_kv_transfer_params,
    resolve_session_id,
};

/// Lowered chat request plus the public response metadata carried by every SSE
/// chunk.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedRequest {
    /// Stable OpenAI-style request ID, reused as the external chat request ID.
    pub request_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Public response rendering options for route-layer helpers.
    pub options: ResponseOptions,
    /// Lowered chat request for `vllm-chat`.
    pub chat_request: ChatRequest,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct ResponseOptions {
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether every streamed chunk should carry cumulative usage.
    pub include_continuous_usage: bool,
    /// Whether the caller requested output logprobs on chat choices.
    pub requested_logprobs: bool,
    /// Number of top logprobs to include for each output token.
    pub output_top_logprobs: i32,
    /// Whether the caller requested top-level prompt logprobs.
    pub include_prompt_logprobs: bool,
    /// Whether to include reasoning content in OpenAI responses.
    pub include_reasoning: bool,
    /// Last assistant-role message content to echo back when `echo=true`.
    pub echo: Option<String>,
    /// Whether to include token IDs alongside generated text.
    pub return_token_ids: bool,
    /// Whether to format logprob tokens as `token_id:{id}`.
    pub return_tokens_as_token_ids: bool,
    /// Whether the request forces one named function tool.
    pub is_named_tool_choice: bool,
}

/// Validate and lower one OpenAI chat completion request into the internal chat
/// format.
///
/// `lora_resolution.model_names` must be non-empty; the first entry is used as
/// the base `model` field in responses when no LoRA adapter is selected.
pub(super) fn prepare_chat_request(
    request: ChatCompletionRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;

    let prompt_truncation = resolve_generation_prompt_truncation(
        request.truncate_prompt_tokens,
        request.truncation_side,
    )
    .map_err(|error| text_submit_error("invalid prompt truncation", error))?;

    let request_id = format!("chatcmpl-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());
    let include_reasoning = request.include_reasoning;
    let echo = request
        .echo
        .then(|| extract_last_assistant_content(&request.messages))
        .flatten();
    let messages: Vec<_> = request.messages.into_iter().map(convert_message).try_collect()?;
    let generation_prompt_mode = normalize_generation_prompt_mode(
        request.add_generation_prompt,
        request.continue_final_message,
        &messages,
    )?;

    let template_kwargs = request.chat_template_kwargs.unwrap_or_default();
    let response_format =
        request.response_format.as_ref().map(serde_json::to_value).transpose().map_err(
            |error| {
                ApiError::invalid_request(
                    format!("failed to serialize response_format: {error}"),
                    Some("response_format"),
                )
            },
        )?;

    let include_usage = (request.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);
    let requested_logprobs = request.logprobs;
    let is_named_tool_choice = matches!(&request.tool_choice, Some(ToolChoice::Function { .. }));

    // Auto-enable prompt logprobs for non-streaming echo, matching Python vLLM's
    // behavior.
    let top_logprobs = request.top_logprobs.unwrap_or(0);
    let prompt_logprobs = request
        .prompt_logprobs
        .or((request.echo && !request.stream).then_some(top_logprobs));
    let include_prompt_logprobs = prompt_logprobs.is_some();

    let structured_outputs = convert_from_response_format(
        request.response_format.as_ref(),
        &request.structured_outputs,
    )?;
    let session_id = resolve_session_id(
        &ctx,
        request.session_id.as_deref(),
        request.vllm_xargs.as_ref(),
    );

    let tool_context = ResolvedToolContext::new(
        &messages,
        convert_tools(request.tools)?,
        request.tool_choice.as_ref().map(convert_tool_choice).transpose()?,
        request.parallel_tool_calls.unwrap_or(true),
    )
    .map_err(|error| chat_submit_error("failed to resolve request tools", error))?;

    let chat_request = ChatRequest {
        request_id: request_id.clone(),
        messages,
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: request.seed,
            max_tokens: request.max_completion_tokens,
            min_tokens: request.min_tokens,
            thinking_token_budget: request.thinking_token_budget,
            logprobs: request.logprobs.then_some(top_logprobs),
            prompt_logprobs,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            repetition_detection: request.repetition_detection,
            stop_token_ids: request.stop_token_ids,
            ignore_eos: request.ignore_eos,
            logit_bias: convert_logit_bias(request.logit_bias)?,
            allowed_token_ids: request.allowed_token_ids,
            bad_words: request.bad_words,
            logprob_token_ids: None,
            structured_outputs,
            skip_reading_prefix_cache: None,
            vllm_xargs: merge_kv_transfer_params(
                merge_ec_transfer_params(request.vllm_xargs, request.ec_transfer_params.as_ref()),
                request.kv_transfer_params.as_ref(),
            ),
        },
        chat_options: ChatOptions {
            generation_prompt_mode,
            chat_template: request.chat_template,
            reasoning_effort: request.reasoning_effort,
            response_format,
            template_kwargs,
        },
        tool_context,
        decode_options: vllm_text::output::TextDecodeOptions {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
            stop_strings: request.stop.map(|stop| stop.into_vec()),
            min_tokens: request.min_tokens.unwrap_or(0),
        },
        intermediate: request.stream,
        prompt_truncation,
        priority: ctx.priority.or(request.priority).unwrap_or(0),
        documents: request.documents,
        cache_salt: request.cache_salt,
        add_special_tokens: request.add_special_tokens,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id,
        lora_request: lora_resolution.lora_request.clone(),
    };

    Ok(PreparedRequest {
        request_id,
        response_model,
        options: ResponseOptions {
            include_usage,
            include_continuous_usage,
            requested_logprobs,
            output_top_logprobs: top_logprobs,
            include_prompt_logprobs,
            include_reasoning,
            echo,
            return_token_ids: request.return_token_ids.unwrap_or(false),
            return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
            is_named_tool_choice,
        },
        chat_request,
    })
}
pub(crate) fn normalize_generation_prompt_mode(
    add_generation_prompt: Option<bool>,
    continue_final_message: bool,
    messages: &[VllmChatMessage],
) -> Result<GenerationPromptMode, ApiError> {
    if add_generation_prompt == Some(true) && continue_final_message {
        bail_invalid_request!(
            "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
        );
    }

    let last_role = messages.last().map(VllmChatMessage::role);
    match (add_generation_prompt, continue_final_message, last_role) {
        (Some(true), true, _) => unreachable!("rejected above"),
        (_, true, Some(vllm_chat::ChatRole::Assistant)) => {
            Ok(GenerationPromptMode::ContinueFinalAssistant)
        }
        (_, true, _) => {
            bail_invalid_request!(
                "Cannot set `continue_final_message` to True when the last message is not from the assistant."
            );
        }
        (Some(false), false, _) => Ok(GenerationPromptMode::NoGenerationPrompt),
        (None | Some(true), false, _) => Ok(GenerationPromptMode::StartNewAssistant),
    }
}

/// Extract the text content of the last message if it has the assistant role.
fn extract_last_assistant_content(messages: &[ChatMessage]) -> Option<String> {
    let ChatMessage::Assistant { content, .. } = messages.last()? else {
        return None;
    };
    let text = match content.as_ref()? {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    (!text.is_empty()).then_some(text)
}

/// Lower one OpenAI chat message into the `vllm-chat` message shape.
pub(crate) fn convert_message(message: ChatMessage) -> Result<VllmChatMessage, ApiError> {
    match message {
        ChatMessage::System { content, .. } => {
            Ok(VllmChatMessage::system(convert_content(content)?))
        }
        ChatMessage::User { content, .. } => Ok(VllmChatMessage::user(convert_content(content)?)),
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning,
            name: _,
        } => {
            let mut blocks = Vec::new();
            if let Some(reasoning) = reasoning
                && !reasoning.is_empty()
            {
                blocks.push(AssistantContentBlock::Reasoning { text: reasoning });
            }
            if let Some(content) = content {
                blocks.extend(convert_assistant_text_blocks(content)?);
            }
            if let Some(tool_calls) = tool_calls {
                blocks.extend(convert_assistant_tool_calls(tool_calls)?);
            }
            if blocks.is_empty() {
                bail_invalid_request!(
                    "Assistant messages must contain text, reasoning content, or tool_calls."
                );
            }

            Ok(VllmChatMessage::assistant_blocks(blocks))
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => Ok(VllmChatMessage::tool_response(
            convert_content(content)?,
            tool_call_id,
        )),
        ChatMessage::Function { .. } => {
            bail_invalid_request!("Function messages are not supported.")
        }
        ChatMessage::Developer {
            content,
            tools,
            name: _,
        } => Ok(VllmChatMessage::developer(
            convert_content(content)?,
            convert_message_tools(tools)?,
        )),
    }
}

/// Convert the given OpenAI message content value into the internal format in
/// `vllm-chat`.
fn convert_content(content: MessageContent) -> Result<ChatContent, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(ChatContent::Text(text)),
        MessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(ChatContentPart::text(text)),
                ContentPart::ImageUrl { image_url, uuid } => Ok(ChatContentPart::ImageUrl {
                    image_url: image_url.url,
                    detail: image_url.detail,
                    uuid,
                }),
                ContentPart::VideoUrl { video_url, uuid } => Ok(ChatContentPart::VideoUrl {
                    video_url: video_url.url,
                    uuid,
                }),
                ContentPart::AudioUrl { audio_url, uuid } => Ok(ChatContentPart::AudioUrl {
                    audio_url: audio_url.url,
                    uuid,
                }),
                ContentPart::InputAudio { input_audio, uuid } => Ok(ChatContentPart::InputAudio {
                    data: input_audio.data,
                    format: input_audio.format,
                    uuid,
                }),
            })
            .try_collect()
            .map(ChatContent::Parts),
    }
}

/// Convert the given OpenAI assistant message content into the internal format
/// in `vllm-chat`.
fn convert_assistant_text_blocks(
    content: MessageContent,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(vec![AssistantContentBlock::Text { text }]),
        MessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(AssistantContentBlock::Text { text }),
                _ => bail_invalid_request!(
                    "Only text content parts are supported for assistant messages."
                ),
            })
            .try_collect(),
    }
}

fn convert_assistant_tool_calls(
    tool_calls: Vec<crate::routes::openai::utils::types::ToolCall>,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    tool_calls
        .into_iter()
        .map(|tool_call| {
            if tool_call.tool_type != "function" {
                bail_invalid_request!("Only function tool calls are supported.");
            }

            Ok(AssistantContentBlock::ToolCall(AssistantToolCall {
                id: tool_call.id,
                name: tool_call.function.name,
                arguments: tool_call.function.arguments.unwrap_or_else(|| "{}".to_string()),
            }))
        })
        .collect()
}

pub(crate) fn convert_tools(tools: Option<Vec<Tool>>) -> Result<Vec<ChatTool>, ApiError> {
    tools
        .unwrap_or_default()
        .into_iter()
        .map(|tool| {
            if tool.tool_type != "function" {
                bail_invalid_request!("Only function tools are supported.");
            }
            Ok(ChatTool {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
                strict: tool.function.strict,
            })
        })
        .collect()
}

fn convert_message_tools(tools: Option<Vec<Tool>>) -> Result<Option<Vec<ChatTool>>, ApiError> {
    let tools = convert_tools(tools)?;
    Ok((!tools.is_empty()).then_some(tools))
}

fn convert_tool_choice(tool_choice: &ToolChoice) -> Result<ChatToolChoice, ApiError> {
    match tool_choice {
        ToolChoice::Value(ToolChoiceValue::Auto) => Ok(ChatToolChoice::Auto),
        ToolChoice::Value(ToolChoiceValue::None) => Ok(ChatToolChoice::None),
        ToolChoice::Value(ToolChoiceValue::Required) => Ok(ChatToolChoice::Required),
        ToolChoice::Function {
            tool_type,
            function,
        } if tool_type == "function" => Ok(ChatToolChoice::Function {
            name: function.name.clone(),
        }),
        ToolChoice::AllowedTools { tools, .. } => bail_invalid_request!(
            "allowed_tools tool_choice is not supported yet: {}.",
            tools.iter().map(ToolReference::identifier).join(", ")
        ),
        _ => bail_invalid_request!("tool_choice={:?} is not supported yet.", tool_choice),
    }
}
