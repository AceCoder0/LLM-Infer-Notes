# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a personal notes/knowledge base on LLM inference attention mechanisms, not a software project. There is no build system, test suite, or runnable code. All content is prose, math formulas, and code snippets extracted from external implementations.

## Content structure

- **`Attention/SFA.md`** — Covers two attention mechanisms:
  - **MLA (Multi-head Latent Attention)**: Used in DeepSeek V2/V3/V3.1. Describes KV compression via low-rank latent vectors, Query compression, and the "absorbed" (MQA-style) vs "non-absorbed" (MHA-style) variants.
  - **SFA (Sparse Flash Attention)**: Used in DeepSeek V3.2 and GLM5. Extends MLA with a Lightning Indexer that computes token-level relevance scores and selects Top-K KV entries for sparse attention.
  - Includes annotated code from MindIE-LLM (`mindie_llm/runtime/layers/attention/backend/sparse_attention.py`) and vLLM+Ascend (`vllm_ascend/attention/sfa_v1.py`), both running on Huawei Ascend NPUs.

- **`assets/`** — Diagram images referenced by the markdown (hosted on an external CDN; the local files may be supplementary).

## Conventions

- Chinese is the primary writing language; code comments and variable names in code snippets are in English.
- Math notation uses LaTeX within `$$` blocks.
- Code snippets reference specific file paths in external projects (MindIE-LLM, vLLM-ascend) — this repo does not contain those implementations.
- When adding new notes, follow the existing pattern: formula explanation → implementation details → annotated code snippets.
