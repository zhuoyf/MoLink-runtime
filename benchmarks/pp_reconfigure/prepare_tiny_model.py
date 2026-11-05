# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a deterministic local checkpoint for the real-GPU continuity test."""

import argparse
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--tokenizer", required=True)
args = parser.parse_args()
torch.manual_seed(123)
config = LlamaConfig(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=10,
    num_attention_heads=2,
    num_key_value_heads=2,
    vocab_size=32000,
    max_position_embeddings=512,
)
model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
model.save_pretrained(args.output)
for source in Path(args.tokenizer).glob("tokenizer*"):
    destination = Path(args.output) / source.name
    if not destination.exists():
        destination.symlink_to(source.resolve())
print(f"Saved ten-layer Llama test checkpoint to {args.output}")
