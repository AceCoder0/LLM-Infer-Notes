# KVCache 笔记

> 研究 vLLM / SGLang / MindIE-LLM 中的 KV Cache 管理：分组、打包布局、前缀缓存、PD 分离等。

## 已覆盖

- **`vllm_hybrid_kv_cache.md`** — vLLM 混合 KV Cache 管理（DeepSeek V4 视角）：
  - DSV4 异构 KV Cache 背景（swa / compressed / indexer / compressor 暂存态）
  - vLLM 三大设计决策 + 分组准则（`group_and_unify_kv_cache_specs`）
  - **谁和谁组成一个 slab**（`_get_packed_kv_cache_layout`，含 3 组示例图 + DSV4 Flash 66 tensor 实例）
  - 运行时（一个 backing buffer 错位视图）、动态分配、GPU vs Ascend 差异

## 待研究（roadmap）

以下方向是本目录的后续主线，README 先列骨架，逐步补充：

### 1. Hybrid KV Cache × 模型（DSV4 / Qwen3.5）
- [ ] DSV4 主 kv、indexer kv、compressor 态在 block 层面如何按请求分配 / 释放
- [ ] Qwen3.5 的 KV Cache 结构（是否引入 hybrid？注意力结构？与 DSV4 的异同）
- [ ] 不同压缩比 / 窗口大小的 trade-off（prefix cache hit rate vs 算子性能）

### 2. Hybrid KV Cache × Prefix Caching
- [ ] 前缀命中的 block 复用如何在多组、多 block table 下工作
- [ ] hash 校验的是哪一层的内容；压缩 kv 的前缀能否复用
- [ ] Ascend 上 block_size 偏大对 prefix hit rate 的影响

### 3. Hybrid KV Cache × PD 分离（prefill / decode 分离）
- [ ] prefill 与 decode 的 KV 传输、块迁移
- [ ] packed slab 布局在 PD 分离下的约束（两边的 block table / tensor 布局一致性）

### 4. SGLang 的设计与实现
- [ ] SGLang 的 KV cache 管理器结构（RadixAttention、ChunkCache、PagedAttention）
- [ ] SGLang 如何支持 DSV4 这类异构 cache（与 vLLM packed slab 的对比）
- [ ] SGLang × Ascend 后端（`srt/hardware_backend/`）的 KV cache 处理

### 5. MindIE-LLM 对照
- [ ] LRU / Prefix Cache / CoW 与 vLLM 方案的映射

## 约定

- 中文为主，代码注释保留英文；
- 每个主题：机制说明 → 关键代码（附外部仓库路径与行号）→ 图示/示例；
- 行号以核对时的代码为准，可能随上游演进漂移。
