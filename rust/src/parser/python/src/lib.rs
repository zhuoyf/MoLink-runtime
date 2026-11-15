// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Thin PyO3 bindings for `vllm_parser::tool`.
//!
//! This crate exposes the Rust tool parser trait and data shapes to Python
//! while keeping parser state, grammar, and schema-aware argument conversion in
//! Rust. Python callers should use this module as a typed bridge and keep any
//! vLLM protocol adaptation outside the binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pythonize::{depythonize, pythonize};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_parser::tool::{Tool, ToolCallDelta, ToolParser, ToolParserOutput};

macro_rules! tool_parser_factory {
    ($($parser:ident),+ $(,)?) => {
        fn create_tool_parser(
            name: &str,
            tools: &[Tool],
        ) -> PyResult<Box<dyn ToolParser>> {
            match name {
                $(
                    stringify!($parser) => {
                        <vllm_parser::tool::$parser as ToolParser>::create(tools)
                    }
                )+
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported tool parser `{name}`"
                    )));
                }
            }
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
        }
    };
}

// Export a tool parser to Python by registering it here.
tool_parser_factory! {
    MinimaxM3ToolParser,

    // Below are the parsers just for testing purposes on Python side.
    DeepSeekV4ToolParser,
    KimiK2ToolParser,
}

#[pyclass(name = "Tool", module = "vllm._rust_tool_parser", skip_from_py_object)]
#[derive(Clone)]
struct PyTool(Tool);

#[pymethods]
impl PyTool {
    #[new]
    #[pyo3(signature = (name, description, parameters, strict=None))]
    fn new(
        name: String,
        description: Option<String>,
        parameters: &Bound<'_, PyAny>,
        strict: Option<bool>,
    ) -> PyResult<Self> {
        let parameters = depythonize::<Value>(parameters).map_err(|error| {
            PyValueError::new_err(format!(
                "failed to convert tool parameters from Python to JSON: {error}"
            ))
        })?;
        Ok(Self(Tool {
            name,
            description,
            parameters,
            strict,
        }))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.0.name
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.0.description.as_deref()
    }

    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pythonize(py, &self.0.parameters).map(Bound::unbind).map_err(|error| {
            PyValueError::new_err(format!(
                "failed to convert tool parameters from JSON to Python: {error}"
            ))
        })
    }

    #[getter]
    fn strict(&self) -> Option<bool> {
        self.0.strict
    }
}

#[pyclass(
    name = "ToolCallDelta",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyToolCallDelta(ToolCallDelta);

#[pymethods]
impl PyToolCallDelta {
    #[new]
    #[pyo3(signature = (tool_index, name, arguments))]
    fn new(tool_index: usize, name: Option<String>, arguments: String) -> Self {
        Self(ToolCallDelta {
            tool_index,
            name,
            arguments,
        })
    }

    #[getter]
    fn tool_index(&self) -> usize {
        self.0.tool_index
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    #[getter]
    fn arguments(&self) -> &str {
        &self.0.arguments
    }
}

#[pyclass(
    name = "ToolParserOutput",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyToolParserOutput(ToolParserOutput);

#[pymethods]
impl PyToolParserOutput {
    #[new]
    #[pyo3(signature = (normal_text="", calls=None))]
    fn new(py: Python<'_>, normal_text: &str, calls: Option<Vec<Py<PyToolCallDelta>>>) -> Self {
        let mut output = ToolParserOutput::default();
        output.push_text(normal_text);
        for call in calls.unwrap_or_default() {
            output.push_call(call.borrow(py).0.clone());
        }
        Self(output)
    }

    #[getter]
    fn normal_text(&self) -> String {
        self.0.normal_text()
    }

    #[getter]
    fn calls(&self) -> Vec<PyToolCallDelta> {
        self.0.calls().into_iter().cloned().map(PyToolCallDelta).collect()
    }

    fn append(&mut self, other: PyRef<'_, PyToolParserOutput>) {
        self.0.append(other.0.clone());
    }

    fn coalesce(&self) -> Self {
        Self(self.0.clone().coalesce())
    }
}

#[pyclass(name = "ToolParser", module = "vllm._rust_tool_parser", unsendable)]
struct PyToolParser(Box<dyn ToolParser>);

impl PyToolParser {
    fn parse_into_output(&mut self, chunk: &str, output: &mut PyToolParserOutput) -> PyResult<()> {
        self.0
            .parse_into(chunk, &mut output.0)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }
}

#[pymethods]
impl PyToolParser {
    #[new]
    fn new(py: Python<'_>, parser_name: &str, tools: Vec<Py<PyTool>>) -> PyResult<Self> {
        let tools = tools.iter().map(|tool| tool.borrow(py).0.clone()).collect::<Vec<_>>();
        create_tool_parser(parser_name, &tools).map(Self)
    }

    fn parse_into(
        &mut self,
        chunk: &str,
        mut output: PyRefMut<'_, PyToolParserOutput>,
    ) -> PyResult<()> {
        self.parse_into_output(chunk, &mut output)
    }

    fn finish(&mut self) -> PyResult<PyToolParserOutput> {
        self.0
            .finish()
            .map(PyToolParserOutput)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }

    fn preserve_special_tokens(&self) -> bool {
        self.0.preserve_special_tokens()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.0.tool_call_id(tool_index)
    }
}

#[pymodule]
fn _rust_tool_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTool>()?;
    m.add_class::<PyToolCallDelta>()?;
    m.add_class::<PyToolParserOutput>()?;
    m.add_class::<PyToolParser>()?;
    Ok(())
}
