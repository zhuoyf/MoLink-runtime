// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

/// Effective model dtype reported by the engine after config resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDtype {
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    BFloat16,
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }
}
