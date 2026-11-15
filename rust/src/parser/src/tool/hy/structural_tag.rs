// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structural-tag grammar for HY XML-style tool calls.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::Value;
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{StructuralTagBuilder, StructuralTagContext};
use xgrammar_structural_tag::format::{Format, StructuralTag, TagFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam};

use super::{HyDialect, HyToolMarkers};

/// HY structural-tag builder using tokenizer-specific structural markers.
#[derive(Debug, Clone)]
pub(super) struct HyStructuralTagBuilder {
    markers: Arc<HyToolMarkers>,
    dialect: HyDialect,
}

impl HyStructuralTagBuilder {
    pub(super) fn new(markers: Arc<HyToolMarkers>, dialect: HyDialect) -> Self {
        Self { markers, dialect }
    }

    fn argument_pair(&self, key: &str) -> Format {
        let excludes = self.markers.iter().collect::<Vec<_>>();
        let separator = self.dialect.separator();
        let mut elements = vec![
            Format::const_string(&self.markers.arg_key_start),
            Format::const_string(key),
            Format::const_string(&self.markers.arg_key_end),
        ];
        if !separator.is_empty() {
            elements.push(Format::const_string(separator));
        }
        elements.extend([
            Format::const_string(&self.markers.arg_value_start),
            Format::any_text_excluding(&excludes),
            Format::const_string(&self.markers.arg_value_end),
        ]);
        if !separator.is_empty() {
            elements.push(Format::const_string(separator));
        }
        Format::sequence(elements)
    }

    fn tool_call(&self, tool: &FunctionToolParam) -> TagFormat {
        let (required_keys, optional_keys) = argument_keys(tool.function.parameters.as_ref());
        let mut elements =
            required_keys.into_iter().map(|key| self.argument_pair(key)).collect::<Vec<_>>();

        if !optional_keys.is_empty() {
            let mut pairs =
                optional_keys.into_iter().map(|key| self.argument_pair(key)).collect::<Vec<_>>();
            let optional = if pairs.len() == 1 {
                pairs.pop().unwrap()
            } else {
                Format::or(pairs)
            };
            elements.push(Format::star(optional));
        }

        let content = if elements.is_empty() {
            match self.dialect {
                HyDialect::V3 => Format::any_text(),
                HyDialect::V4 => Format::const_string(""),
            }
        } else {
            Format::sequence(elements)
        };
        let tool_sep = self.markers.tool_sep.as_deref().unwrap_or_default();
        TagFormat::new(
            format!(
                "{}{}{}{}",
                self.markers.tool_call_start,
                tool.function.name,
                tool_sep,
                self.dialect.separator(),
            ),
            content,
            self.markers.tool_call_end.clone(),
        )
    }

    fn tool_calls(&self, tools: &[FunctionToolParam], choice: BuilderToolChoice) -> Format {
        let mut calls = tools.iter().map(|tool| self.tool_call(tool)).collect::<Vec<_>>();
        let separator = self.dialect.separator();
        let begin = format!("{}{separator}", self.markers.tool_calls_start);
        let end = format!("{separator}{}", self.markers.tool_calls_end);

        match choice {
            BuilderToolChoice::Auto if calls.is_empty() => Format::any_text(),
            BuilderToolChoice::Auto => {
                let outer = TagFormat::new(
                    begin,
                    Format::tags_with_separator(calls, separator, true, false),
                    end,
                );
                Format::triggered_tags(&[&self.markers.tool_calls_start], vec![outer])
            }
            BuilderToolChoice::Forced => Format::sequence(vec![
                Format::const_string(begin),
                Format::Tag(calls.pop().unwrap()),
                Format::const_string(end),
            ]),
            BuilderToolChoice::Required => Format::sequence(vec![
                Format::const_string(begin),
                Format::tags_with_separator(calls, separator, true, false),
                Format::const_string(end),
            ]),
        }
    }
}

impl StructuralTagBuilder for HyStructuralTagBuilder {
    fn build(&self, ctx: StructuralTagContext<'_>) -> Result<StructuralTag> {
        Ok(StructuralTag::new(
            self.tool_calls(ctx.function_tools, ctx.tool_choice),
        ))
    }
}

/// Split argument keys into required and optional declaration-order groups.
fn argument_keys(parameters: Option<&Value>) -> (Vec<&str>, Vec<&str>) {
    let Some(parameters) = parameters.and_then(Value::as_object) else {
        return (Vec::new(), Vec::new());
    };
    let properties = parameters.get("properties").and_then(Value::as_object);
    let required = parameters
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let required_set = required.iter().copied().collect::<HashSet<_>>();

    let mut required_keys = properties
        .into_iter()
        .flat_map(|properties| properties.keys())
        .map(String::as_str)
        .filter(|key| required_set.contains(key))
        .collect::<Vec<_>>();
    required_keys.extend(
        required
            .into_iter()
            .filter(|key| properties.is_none_or(|properties| !properties.contains_key(*key))),
    );
    let optional_keys = properties
        .into_iter()
        .flat_map(|properties| properties.keys())
        .map(String::as_str)
        .filter(|key| !required_set.contains(key))
        .collect();
    (required_keys, optional_keys)
}
