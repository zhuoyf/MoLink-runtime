// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Applies xgrammar structural-tag constraints for strict tool calling.

use thiserror_ext::AsReport;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputsParams,
};
use vllm_parser::tool::StructuralTagBuilder;
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_structural_tag,
};

use crate::request::{ChatRequest, ChatToolChoice};
use crate::{Error, Result as ChatResult};

/// Apply structural tag constraints to the request based on the tool parser's structural tag
/// support and the request's tool choice.
pub(super) fn apply_structural_tag_constraint(
    request: &mut ChatRequest,
    builder: Option<&dyn StructuralTagBuilder>,
) -> ChatResult<()> {
    let Some(builder) = builder else {
        return Ok(());
    };
    let Some(tool_choice) = structural_tag_tool_choice(request) else {
        return Ok(());
    };

    let tools = request
        .tools()
        .iter()
        .map(|tool| {
            ToolParam::Function(FunctionToolParam::new(FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.parameters.clone()),
                strict: tool.strict,
            }))
        })
        .collect::<Vec<_>>();

    let structural_tag = build_structural_tag(
        builder,
        &tools,
        tool_choice,
        StructuralTagOptions::default().with_reasoning(false),
    )
    .and_then(|tag| tag.to_json_string())
    .map_err(|error| Error::StructuralTag {
        message: error.to_report_string(),
    })?;

    // Overwrite any existing structured output settings with the structural tag constraint.
    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        backend: StructuredOutputBackend::Xgrammar,
        ..StructuredOutputsParams::structural_tag(structural_tag)
    });

    Ok(())
}

/// Resolve the tool choice used for [`xgrammar_structural_tag`] based on the request.
///
/// Returns `None` if no structural tag constraints should be applied.
fn structural_tag_tool_choice(request: &ChatRequest) -> Option<StructuralTagToolChoice> {
    if request.tools().is_empty() {
        return None;
    }

    match request.tool_choice() {
        // For `Auto`, only apply the structural tag if there's at least one strict tool.
        ChatToolChoice::Auto if request.tools().iter().any(|tool| tool.strict == Some(true)) => {
            Some(StructuralTagToolChoice::auto())
        }
        ChatToolChoice::Auto | ChatToolChoice::None => None,

        ChatToolChoice::Required => Some(StructuralTagToolChoice::required()),
        ChatToolChoice::Function { name } => Some(StructuralTagToolChoice::function(name.clone())),
    }
}
