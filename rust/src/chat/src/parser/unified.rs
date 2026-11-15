// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser registration and selection boundary for `vllm-chat`.

use std::sync::LazyLock;

pub use vllm_parser::unified::{
    Gemma4UnifiedParser, HyV3UnifiedParser, HyV4UnifiedParser, InklingUnifiedParser,
    KimiK3UnifiedParser, UnifiedParser,
};
use vllm_tokenizer::DynTokenizer;

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered unified parsers.
pub mod names {
    pub const GEMMA4: &str = "gemma4";
    pub const HY_V3: &str = "hy_v3";
    pub const HY_V4: &str = "hy_v4";
    pub const INKLING: &str = "inkling";
    pub const KIMI_K3: &str = "kimi_k3";
}

/// Constructor signature for one registered unified parser implementation.
type UnifiedParserCreator =
    fn(&[ChatTool], DynTokenizer) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>>;

/// Registry and model matcher for unified parsers.
pub type UnifiedParserFactory = ParserFactory<UnifiedParserCreator>;

impl UnifiedParserFactory {
    /// Get the global unified parser factory with built-in registrations and
    /// model mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<UnifiedParserFactory> = LazyLock::new(UnifiedParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory.register_parser::<Gemma4UnifiedParser>(names::GEMMA4);
        factory.register_parser::<HyV3UnifiedParser>(names::HY_V3);
        factory.register_parser::<HyV4UnifiedParser>(names::HY_V4);
        factory.register_parser::<InklingUnifiedParser>(names::INKLING);
        factory.register_parser::<KimiK3UnifiedParser>(names::KIMI_K3);

        factory
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("hy3", names::HY_V3)
            .register_pattern("hy4", names::HY_V4)
            .register_pattern("inkling", names::INKLING)
            .register_pattern("kimi-k3", names::KIMI_K3)
            .register_pattern("kimi_k3", names::KIMI_K3);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: UnifiedParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tools: &[ChatTool],
        tokenizer: DynTokenizer,
    ) -> crate::Result<Box<dyn UnifiedParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "unified",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tools, tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "unified",
            name: name.to_string(),
            error: error.into(),
        })
    }
}
