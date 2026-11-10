# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit request conflicts, preserved across migration RPC boundaries."""


class ReconfigurationValidationError(ValueError):
    pass
