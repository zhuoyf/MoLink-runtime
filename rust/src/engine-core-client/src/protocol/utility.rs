// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::any::type_name;
use std::fmt;
use std::str::FromStr;

use rmpv::Value;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use serde_with::{DeserializeFromStr, SerializeDisplay};
use thiserror_ext::AsReport;

use super::{OpaqueValue, default_opaque_value_nil};
use crate::error::{Error, Result};

/// How pause/sleep utility calls handle in-flight requests.
///
/// Use display/from-str serde so MessagePack utility args stay as Python
/// literal strings instead of serde enum variant tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, SerializeDisplay, DeserializeFromStr)]
pub enum PauseMode {
    /// Abort all in-flight requests immediately.
    #[default]
    Abort,
    /// Wait for in-flight requests to complete.
    Wait,
    /// Freeze queued requests so they can resume later.
    Keep,
}

impl PauseMode {
    /// Return the Python literal used on the utility-call wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Abort => "abort",
            Self::Wait => "wait",
            Self::Keep => "keep",
        }
    }
}

impl fmt::Display for PauseMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PauseMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "abort" => Ok(Self::Abort),
            "wait" => Ok(Self::Wait),
            "keep" => Ok(Self::Keep),
            other => Err(format!(
                "invalid pause mode `{other}`; expected one of: abort, wait, keep"
            )),
        }
    }
}

/// Utility call id as carried on the engine-core MessagePack wire.
///
/// Python emits utility ids as MessagePack integers, including values that may
/// require unsigned 64-bit encoding. Keep MessagePack's signed/unsigned
/// integer distinction instead of flattening to `i64` or `u64` at decode time.
#[derive(Clone, Copy, PartialEq)]
pub struct UtilityCallId(rmpv::Integer);

impl UtilityCallId {
    /// Returns the integer represented as `u64` if possible, or else `None`.
    /// This is the typical case for utility calls.
    pub fn as_u64(self) -> Option<u64> {
        self.0.as_u64()
    }

    /// Returns the integer represented as `i64` if possible, or else `None`.
    pub fn as_i64(self) -> Option<i64> {
        self.0.as_i64()
    }
}

impl Default for UtilityCallId {
    fn default() -> Self {
        Self(0_u64.into())
    }
}

impl From<u64> for UtilityCallId {
    fn from(value: u64) -> Self {
        Self(value.into())
    }
}

impl TryFrom<Value> for UtilityCallId {
    type Error = String;

    fn try_from(value: Value) -> std::result::Result<Self, Self::Error> {
        match value {
            Value::Integer(value) => Ok(UtilityCallId(value)),
            other => Err(format!(
                "expected a MessagePack integer utility call id, got {other}"
            )),
        }
    }
}

impl PartialEq<u64> for UtilityCallId {
    fn eq(&self, other: &u64) -> bool {
        self.as_u64() == Some(*other)
    }
}

impl fmt::Debug for UtilityCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Display for UtilityCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Serialize for UtilityCallId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Value::Integer(self.0).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UtilityCallId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Integer(value) => Ok(UtilityCallId(value)),
            other => Err(serde::de::Error::custom(format!(
                "expected a MessagePack integer utility call id, got {other}"
            ))),
        }
    }
}

/// Engine-core utility call payload sent from frontend to engine.
///
/// Original Python payload shape:
/// `(client_index, call_id, method_name, args)`
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct EngineCoreUtilityRequest {
    pub client_index: u32,
    pub call_id: UtilityCallId,
    pub method_name: String,
    pub args: OpaqueValue,
}

impl EngineCoreUtilityRequest {
    /// Create a new utility request with the given strongly typed arguments,
    /// encoding them into the expected msgpack value format.
    pub fn new<T>(
        client_index: u32,
        call_id: u64,
        method_name: impl Into<String>,
        args: T,
    ) -> Result<Self>
    where
        T: Serialize + std::fmt::Debug,
    {
        let args = rmpv::ext::to_value(&args).map_err(|error| Error::Encode {
            target_type: type_name::<T>(),
            message: format!(
                "failed to encode utility args `{args:?}`: {}",
                error.to_report_string()
            ),
        })?;
        let args = match args {
            Value::Nil => Value::Array(Vec::new()),
            other => other,
        };

        Ok(Self {
            client_index,
            call_id: UtilityCallId::from(call_id),
            method_name: method_name.into(),
            args,
        })
    }
}

/// Result of a utility call.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L174-L183>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct UtilityOutput {
    pub call_id: UtilityCallId,
    /// Non-`None` implies the call failed and `result` should be ignored.
    #[serde(default)]
    pub failure_message: Option<String>,
    #[serde(default)]
    pub result: Option<UtilityResultEnvelope>,
}

/// Python `UtilityResult` wrapper carried inside `UtilityOutput.result`.
///
/// Upstream reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/serial_utils.py#L178-L185>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct UtilityResultEnvelope {
    /// Recursive type information encoded on Python side, serving as the hint
    /// for deserialization. We don't care it here as in Rust frontend all
    /// utility calls are strongly-typed.
    #[serde(default)]
    type_info: Option<OpaqueValue>,
    /// The actual utility result.
    #[serde(default = "default_opaque_value_nil")]
    result: OpaqueValue,
}

impl UtilityResultEnvelope {
    /// Create a utility result envelope without type information.
    pub fn without_type_info(result: OpaqueValue) -> Self {
        Self {
            type_info: None,
            result,
        }
    }
}

impl UtilityOutput {
    /// Decode the typed result of a utility call.
    pub fn into_typed_result<T>(self, method: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        if let Some(message) = self.failure_message {
            return Err(Error::UtilityCallFailed {
                method: method.to_string(),
                call_id: self.call_id,
                message,
            });
        }

        let result = self.result.map(|e| e.result).unwrap_or(Value::Nil);

        rmpv::ext::from_value(result).map_err(|error| Error::UtilityResultDecode {
            method: method.to_string(),
            call_id: self.call_id,
            message: error.to_report_string(),
        })
    }
}
