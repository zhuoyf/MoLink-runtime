# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducible GPU compute load for the adaptive scheduler experiment."""

import argparse
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--seconds", type=float, default=20)
args = parser.parse_args()
torch.accelerator.set_device_index(args.gpu)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = torch.empty_like(x)
end = time.monotonic() + args.seconds
while time.monotonic() < end:
    torch.mm(x, x, out=y)
    torch.accelerator.synchronize()
