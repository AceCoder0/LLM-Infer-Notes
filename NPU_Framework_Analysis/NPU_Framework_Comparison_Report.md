# SGLang vs MindIE-LLM vs vLLM-Ascend NPU 推理框架对比分析报告

## 目录

1. [框架概述](#1-框架概述)
2. [架构设计对比](#2-架构设计对比)
3. [Attention 机制实现](#3-attention-机制实现)
4. [KV Cache 管理](#4-kv-cache-管理)
5. [调度器设计](#5-调度器设计)
6. [分布式与并行策略](#6-分布式与并行策略)
7. [量化和压缩支持](#7-量化和压缩支持)
8. [推测解码支持](#8-推测解码支持)
9. [接口与易用性](#9-接口与易用性)
10. [总结对比表](#10-总结对比表)

---

## 1. 框架概述

### 1.1 SGLang NPU

**定位**：SGLang 是一个上层推理框架，通过 RadixAttention 实现高效的注意力机制和 KV Cache 复用。SGLang 对 NPU 的支持是其多硬件后端的一部分。

**代码组织**：
```
sglang/python/sglang/srt/hardware_backend/npu/
├── attention/                    # 注意力后端实现
│   ├── ascend_backend.py         # 主attention后端
│   ├── ascend_torch_native_backend.py
│   ├── ascend_gdn_backend.py
│   ├── mla_preprocess.py
├── graph_runner/                 # CUDA Graph支持
│   ├── npu_graph_runner.py
│   ├── eagle_draft_npu_graph_runner.py
├── modules/                      # NPU特定模块
├── quantization/                 # 量化实现
├── memory_pool_npu.py           # 内存池管理
├── allocator_npu.py             # 分页分配器
└── utils.py
```

**核心技术栈**：
- PyTorch + torch_npu (Ascend PyTorch适配)
- CANN (Compute Architecture for Neural Networks) 8.5.0
- Triton-ascend (定制Triton)
- SGL-Kernel NPU (自定义算子)

### 1.2 vLLM-Ascend

**定位**：vLLM-Ascend 是 vLLM 项目的官方昇腾后端插件，通过硬件插件机制无缝集成到 vLLM 架构中。

**代码组织**：
```
vllm-ascend/vllm_ascend/
├── attention/                    # 注意力实现
│   ├── attention_v1.py          # v1版本attention后端
│   ├── mla_v1.py                # MLA支持
│   ├── sfa_v1.py                # SFA支持
│   └── context_parallel/        # 上下文并行
├── worker/
│   ├── model_runner_v1.py       # 模型运行器
│   ├── model_runner_310p.py    # 310P特定
├── compilation/
│   ├── acl_graph.py            # 图编译支持
├── ops/                         # 算子实现
├── quantization/                # 量化方法
├── distributed/                 # 分布式通信
└── spec_decode/                 # 推测解码
```

**核心技术栈**：
- 基于 vLLM v1 版本架构
- torch_npu 算子
- NPUGraph (图执行优化)
- 310P 专用实现

### 1.3 MindIE-LLM

**定位**：MindIE-LLM 是昇腾原厂的大模型推理加速套件，采用 C++ 核心引擎 + Python 接口的设计，提供企业级推理能力。

**代码组织**：
```
MindIE-LLM/
├── src/                         # C++ 核心引擎
│   ├── engine/                  # 推理引擎主逻辑
│   │   ├── llm_engine.h/cpp
│   │   └── construct_execute_request.cpp
│   ├── scheduler/               # 调度器
│   │   ├── scheduler.h/cpp
│   │   └── policy/              # 调度策略
│   ├── block_manager/           # KV Cache块管理
│   │   ├── self_attn_block_manager.cpp
│   │   └── prefix_cache_block.cpp
│   ├── llm_manager/             # Python/C++桥接
│   ├── server/                  # 服务端点
│   │   ├── endpoint/           # HTTP/gRPC接口
│   │   └── single_req_infer_interface/
│   └── executor/                # 执行器通信
├── mindie_llm/                  # Python推理框架
│   ├── connector/               # 请求接入
│   ├── text_generator/          # 核心推理引擎
│   ├── modeling/                # 模型封装
│   └── runtime/                 # 运行时
└── examples/                   # 示例代码
```

**核心技术栈**：
- C++ 核心引擎 (高并发、低延迟)
- Python 接口封装
- 深度优化算子库
- 多级调度策略

---

## 2. 架构设计对比

### 2.1 整体架构

| 特性 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|------|------------|-------------|------------|
| 架构类型 | Python-first, RadixAttention | Python + 插件机制 | C++核心 + Python桥接 |
| 核心语言 | Python | Python + C++扩展 | C++ + Python绑定 |
| 设计理念 | 轻量级、灵活性 | 通用、可扩展 | 企业级、高性能 |
| 后端绑定 | 深度绑定NPU | 解耦的插件架构 | 深度绑定昇腾 |

### 2.2 分层架构对比

#### SGLang NPU 分层

```
┌─────────────────────────────────────────┐
│         上层接口 (Python)                │
│  launch_server / OpenAI API / REST      │
├─────────────────────────────────────────┤
│         调度器 (Scheduler)               │
│  - ForwardMode (extend/decode/...)      │
│  - BatchScheduler                       │
├─────────────────────────────────────────┤
│       RadixAttention + AttentionBackend │
│  - AscendAttnBackend                    │
│  - Prefix Cache                         │
├─────────────────────────────────────────┤
│       Model Executor                    │
│  - ModelRunner                          │
│  - TokenToKVPool                        │
├─────────────────────────────────────────┤
│       NPU Backend                       │
│  - torch_npu / CANN                     │
│  - SGL-Kernel NPU                       │
│  - Triton-ascend                        │
└─────────────────────────────────────────┘
```

#### vLLM-Ascend 分层

```
┌─────────────────────────────────────────┐
│         上层接口 (Python)                │
│  LLM / AsyncLLM / OpenAI API            │
├─────────────────────────────────────────┤
│       ModelRunner (v1/v2)               │
│  - SchedulerOutput                      │
│  - SamplingMetadata                      │
├─────────────────────────────────────────┤
│       Attention Backend (插件化)          │
│  - AscendAttentionBackend               │
│  - AttentionMetadataBuilder              │
├─────────────────────────────────────────┤
│       算子层                             │
│  - torch_npu._npu_paged_attention       │
│  - torch_npu.npu_fused_infer_attention  │
├─────────────────────────────────────────┤
│       编译层                             │
│  - NPUGraph / ACL Graph                 │
└─────────────────────────────────────────┘
```

#### MindIE-LLM 分层

```
┌─────────────────────────────────────────┐
│         服务端点 (C++)                   │
│  HTTP/gRPC / single_req_infer_interface │
├─────────────────────────────────────────┤
│         LLM Manager (Python/C++)        │
│  - 请求路由                              │
│  - 会话管理                              │
├─────────────────────────────────────────┤
│         LLM Engine (C++)                │
│  - SchedulerThread                      │
│  - ScheduleExecTransfer                 │
├─────────────────────────────────────────┤
│         Scheduler (C++)                 │
│  - FCFS / PDDS / Layerwise              │
│  - DynamicBatchSize                     │
├─────────────────────────────────────────┤
│         Block Manager (C++)              │
│  - LRU Evictor                          │
│  - Prefix Cache                         │
├─────────────────────────────────────────┤
│         执行器 (Executor)                │
│  - Model Execution                      │
└─────────────────────────────────────────┘
```

### 2.3 设计哲学差异

**SGLang NPU**：
- **RadixAttention 优先**：通过基数树实现 KV Cache 的高效复用
- **Python 原生**：大部分逻辑在 Python 中，便于快速迭代
- **前端友好**：提供丰富的 OpenAI 兼容接口

**vLLM-Ascend**：
- **插件化设计**：通过 `register_backend` 机制支持多种硬件
- **v1/v2 双轨并行**：保持向后兼容的同时推进新架构
- **图编译优化**：NPUGraph 加速计算图执行

**MindIE-LLM**：
- **性能至上**：C++ 核心追求极致性能
- **企业特性**：多级调度、Latency Predictor
- **昇腾深度优化**：与硬件紧密结合

---

## 3. Attention 机制实现

### 3.1 支持的 Attention 类型

| Attention 类型 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|---------------|------------|-------------|------------|
| Vanilla Attention | ✅ | ✅ | ✅ |
| PagedAttention | ✅ | ✅ | ✅ |
| MLA (Multi-head Latent Attention) | ✅ | ✅ | ✅ |
| GQA (Grouped Query Attention) | ✅ | ✅ | ✅ |
| MQA (Multi-query Attention) | ✅ | ✅ | ✅ |
| Ring Attention | ✅ | ✅ | 待查 |
| Flash Attention | ✅ | ✅ | ✅ |
| Streaming Attention (Sinks) | ✅ | 待查 | 待查 |

### 3.2 SGLang NPU Attention 实现

**文件**：[ascend_backend.py](file:///Users/hellozkr/repos/sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py)

**核心类**：`AscendAttnBackend`

**关键特性**：

1. **Forward 模式分发**：
```python
def forward_extend(...):    # Prefill 阶段
def forward_decode(...):    # Decode 阶段
def forward_mtp(...):       # 多步推测
def forward_mixed(...):     # 混合 Chunk
```

2. **FIA (Flash Infer Attention)**：
```python
if self.use_fia:
    # 使用 npu_fused_infer_attention_score
    torch.ops.npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=layer.tp_q_head_num,
        input_layout="BSND",
        sparse_mode=3,
        scale=layer.scaling,
    )
```

3. **MLA 支持**：
```python
if self.use_mla:
    # 使用 npu_ring_mla 或 npu_fused_infer_attention_score
    torch_npu.atb.npu_ring_mla(
        q_nope=q_nope, q_rope=q_rope,
        k_nope=k_nope, k_rope=k_rope,
        value=v, mask=self.ringmla_mask,
        seqlen=seq_lens, ...
    )
```

4. **Sparse Attention (NSA/NPA)**：
```python
def forward_sparse(...):
    # 使用 npu_sparse_flash_attention
    torch_npu.npu_sparse_flash_attention(
        query=q_nope, key=k_nope, value=k_nope,
        query_rope=q_pe, key_rope=k_pe,
        sparse_indices=topk_indices,
        sparse_mode=3, attention_mode=2,
    )
```

### 3.3 vLLM-Ascend Attention 实现

**文件**：[attention_v1.py](file:///Users/hellozkr/repos/vllm-ascend/vllm_ascend/attention/attention_v1.py)

**核心类**：`AscendAttentionBackendImpl`

**关键特性**：

1. **状态机管理**：
```python
class AscendAttentionState(Enum):
    PrefillNoCache = 0      # 无缓存预填充
    PrefillCacheHit = 1    # 缓存命中预填充
    DecodeOnly = 2          # 仅解码
    ChunkedPrefill = 3      # 分块预填充
    SpecDecoding = 4        # 推测解码
```

2. **双路径 Forward**：
```python
def forward_fused_infer_attention(...)  # FIA 路径
def forward_paged_attention(...)        # PA 路径
```

3. **C8 量化支持**：
```python
class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    # INT8 KV Cache 支持
    def _forward_c8_decode(...):
        torch_npu.npu_fused_infer_attention_score(
            query, key, value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            input_layout="BNSD",
            sparse_mode=0,
        )
```

4. **Context Parallel 支持**：
```python
# vllm_ascend/attention/context_parallel/attention_cp.py
class AscendAttentionCPImpl(AttentionImpl):
    # 上下文并行注意力
```

### 3.4 核心 API 对比

| API | SGLang | vLLM-Ascend |
|-----|--------|-------------|
| Flash Attention | `torch_npu.npu_fused_infer_attention_score` | `torch_npu.npu_fused_infer_attention_score` |
| Paged Attention | `torch_npu._npu_paged_attention` | `torch_npu._npu_paged_attention` |
| MLA | `torch_npu.atb.npu_ring_mla` | `npu_fused_infer_attention_score` (TND layout) |
| Sparse | `torch_npu.npu_sparse_flash_attention` | 通过 FIA 实现 |

---

## 4. KV Cache 管理

### 4.1 分页管理机制

#### SGLang NPU

**文件**：[allocator_npu.py](file:///Users/hellozkr/repos/sglang/python/sglang/srt/hardware_backend/npu/allocator_npu.py)

**类**：`NPUPagedTokenToKVPoolAllocator`

```python
class NPUPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def alloc_extend(...):   # 扩展分配
    def alloc_decode(...):   # 解码分配
    def free(...):           # 释放
```

**特点**：
- 继承自 `PagedTokenToKVPoolAllocator`
- 支持 prefix caching
- 小批量使用 kernel 优化 (`sgl_kernel_npu.mem_cache.allocator.alloc_extend_kernel`)

#### vLLM-Ascend

**机制**：
- 使用 vLLM v1 的统一 KV Cache 接口
- `block_tables` 管理物理块映射
- 支持 prefix cache block

```python
# vllm_ascend/attention/attention_v1.py
class AscendAttentionMetadataBuilder:
    def build(...):
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
```

#### MindIE-LLM

**文件**：`src/block_manager/`

**类**：
- `SelfAttnBlockManager` - 自注意力块管理
- `PrefixCacheBlockAllocator` - 前缀缓存块分配
- `LRUEvictor` - LRU 驱逐策略

```cpp
// block_manager/self_attn_block_manager.h
class SelfAttnBlockManager : public BlockManager {
    BlockSpaceManagerSPtr blockManager_;
    // 支持 CoW (Copy-on-Write)
    // 支持 Prefix Cache
};
```

### 4.2 Prefix Cache 支持

| 特性 | SGLang | vLLM-Ascend | MindIE-LLM |
|------|--------|-------------|------------|
| Prefix Cache | ✅ RadixAttention | ✅ | ✅ |
| 算法 | 基数树 (Radix Tree) | Hash-based | Hash-based |
| 共享块 | ✅ | ✅ | ✅ |
| 动态更新 | ✅ | ✅ | ✅ |

---

## 5. 调度器设计

### 5.1 SGLang NPU 调度

SGLang 使用 **ForwardMode** 调度范式：

```python
# python/sglang/srt/layers/radix_attention.py
class ForwardMode(Enum):
    EXTEND = "extend"           # Prefill
    DECODE = "decode"           # Decode
    EXTEND_V1 = "extend_v1"
    DRAFT_EXTEND = "draft_extend"
    TARGET_VERIFY = "target_verify"
    IDLE = "idle"
```

**特点**：
- 简洁的枚举驱动
- 与 RadixAttention 深度集成
- 支持 speculative decoding 模式

### 5.2 vLLM-Ascend 调度

vLLM-Ascend 复用 vLLM v1 的调度器：

```python
# vllm/v1/core/sched/scheduler.py
class Scheduler:
    def schedule(self):
        # 1. 分割 decode 和 prefill
        # 2. 应用 chunked prefill
        # 3. 调度预算管理
```

**特点**：
- 成熟的 v1 调度器
- 支持 continuous batching
- 可配置 chunked prefill

### 5.3 MindIE-LLM 调度

**文件**：[scheduler.h](file:///Users/hellozkr/repos/MindIE-LLM_zhaokerui/src/scheduler/scheduler.h)

```cpp
class Scheduler : public IScheduler {
    std::shared_ptr<Policy> prefillPolicy_;   // FCFS
    std::shared_ptr<Policy> decodePolicy_;    // FCFS
    std::shared_ptr<StagePolicy> stagePolicy_;
    std::shared_ptr<DynamicBatchSize> dynamicBatchSize_;
};
```

**调度策略**：
1. **FCFS** - 先来先服务
2. **PDDS** - 预定义调度
3. **Layerwise** - 分层调度

**关键方法**：
```cpp
std::pair<SequenceGroupMetaDatas, SchedulerOutputs> Schedule(bool needSync = false);
std::pair<SequenceGroupMetaDatas, SchedulerKVTransferOutput> ScheduleTransfer();
```

---

## 6. 分布式与并行策略

### 6.1 Tensor Parallelism (TP)

#### SGLang NPU

```python
# 使用 torch_npu 分布式
from sglang.srt.distributed.device_communicators.npu_communicator import NPUCommunicator

class NPUCommunicator:
    def all_reduce(self, tensor):
        # HCCL 通信
    def broadcast(self, tensor):
        # 广播
```

#### vLLM-Ascend

```python
# vllm_ascend/distributed/device_communicators/npu_communicator.py
class NPUCommunicator:
    # 基于 PyHCCL 的集合通信
```

#### MindIE-LLM

```cpp
// src/executor/communicator.h
class Communicator {
    void AllReduce(Tensor& tensor);
    void Broadcast(Tensor& tensor, int root);
};
```

### 6.2 Expert Parallelism (EP)

| 框架 | MoE 支持 | EP 实现 |
|------|---------|---------|
| SGLang | ✅ | DeepEP 兼容库 |
| vLLM-Ascend | ✅ | EPLB (Expert Parallel Load Balancer) |
| MindIE-LLM | ✅ | 原生支持 |

### 6.3 KV Cache Transfer

**SGLang** - PD Disaggregation:
```bash
# 使用 MemFabric-Hybrid
export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
python -m sglang.launch_server --disaggregation-mode prefill
```

**vLLM-Ascend**:
```python
# vllm_ascend/distributed/kv_transfer/
class AscendMultiConnector:
    # 支持 Mooncake、Ascend Store
```

**MindIE-LLM**:
```cpp
// 边云场景支持
class LayerwiseMixin {
    // 支持 layerwise disaggregated 推理
};
```

---

## 7. 量化和压缩支持

### 7.1 量化方法对比

| 量化方法 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|---------|------------|-------------|------------|
| FP16/BF16 | ✅ | ✅ | ✅ |
| INT8 | ✅ | ✅ | ✅ |
| INT4 | ✅ (W4A4) | ✅ (W4A16) | ✅ |
| W8A8 | ✅ | ✅ | ✅ |
| W8A8 Dynamic | ✅ | ✅ | ✅ |
| FP8 | ✅ | ✅ | 待查 |
| GPTQ | ✅ | ✅ | 待查 |
| GGUF | ✅ | ✅ | 待查 |
| AWQ | ✅ | ✅ | 待查 |
| C8 (INT8 KV Cache) | 待查 | ✅ | 待查 |

### 7.2 SGLang NPU 量化

**文件**：`sglang/python/sglang/srt/hardware_backend/npu/quantization/`

```python
# linear_method_npu.py
class W8A8DynamicLinearMethod:
    # 动态 INT8 量化

# fused_moe_method_npu.py
class FusedMoEMethodNPU:
    # MoE 专用量化
```

### 7.3 vLLM-Ascend 量化

**文件**：`vllm-ascend/vllm_ascend/quantization/methods/`

```python
# w8a8_dynamic.py
class W8A8DynamicMethod:
    # W8A8 动态量化

# kv_c8.py
class KVCacheC8QuantMethod:
    # C8 INT8 KV Cache 量化
    # 支持 antiquant
```

---

## 8. 推测解码支持

### 8.1 SGLang NPU 推测解码

**支持的方法**：
- **Eagle** (Draft Verify)
- **Drafter** (Multi-step)
- **Medusa**
- **N-gram**

**实现文件**：
- `sglang/python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py`
- `sglang/python/sglang/srt/speculative/`

```python
class AscendAttnMultiStepDraftBackend:
    def __init__(self, model_runner, topk, speculative_num_steps):
        self.attn_backends = []
        for step_id in range(speculative_num_steps):
            self.attn_backends.append(
                AscendAttnBackend(model_runner, speculative_step_id=step_id)
            )
```

### 8.2 vLLM-Ascend 推测解码

**文件**：`vllm-ascend/vllm_ascend/spec_decode/`

```python
class AscendEagleProposer(...)
class AscendDflashProposer(...)
class AscendMedusaProposer(...)
```

**支持配置**：
```python
# NPUGraph 推测解码支持
if _EXTRA_CTX.is_draft_model:
    graph_params = get_draft_graph_params()
```

### 8.3 MindIE-LLM 推测解码

**C++ 实现**：
- 支持推测解码机制
- 与调度器深度集成

---

## 9. 接口与易用性

### 9.1 API 接口

#### SGLang NPU

```python
# 启动服务
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend ascend \
    --device npu

# OpenAI 兼容 API
from sglang import OpenAI

client = OpenAI(base_url="http://localhost:8000")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### vLLM-Ascend

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    device="npu",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=2
)

# OpenAI 兼容
from vllm import SamplingParams
```

#### MindIE-LLM

```python
from mindie_llm import TextGenerator

generator = TextGenerator(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    device="npu"
)
response = generator.generate("Hello!")
```

### 9.2 配置文件

| 框架 | 配置方式 | 关键参数 |
|------|---------|---------|
| SGLang | CLI + 环境变量 | `--tp-size`, `--attention-backend`, `--enable-torch-compile` |
| vLLM-Ascend | Python API + envs.py | `tensor_parallel_size`, `gpu_memory_utilization` |
| MindIE-LLM | JSON 配置 + API | `scheduler_config`, `model_config` |

---

## 10. 总结对比表

### 10.1 核心架构

| 维度 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|------|------------|-------------|------------|
| **架构风格** | Python-first, RadixAttention | 插件化, vLLM扩展 | C++核心, 企业级 |
| **核心语言** | Python | Python + C++ | C++ + Python |
| **学习曲线** | 中等 | 较高 | 较高 |
| **定制难度** | 简单 | 中等 | 较难 |
| **社区活跃度** | 活跃 | 活跃 | 官方维护 |

### 10.2 性能特性

| 维度 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|------|------------|-------------|------------|
| **Prefill 优化** | RadixAttention | Chunked Prefill | Dynamic Batch |
| **Decode 优化** | PagedAttention | PagedAttention | Streaming |
| **图编译** | Torch Compile | NPUGraph | 编译优化 |
| **通信优化** | HCCL | PyHCCL | HCCL/自定义 |

### 10.3 功能完整性

| 特性 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|------|------------|-------------|------------|
| **模型支持** | 广泛 | 广泛 | 官方优化 |
| **多模态** | ✅ | ✅ | ✅ |
| **推测解码** | ✅ (多方法) | ✅ | ✅ |
| **LoRA** | ✅ | ✅ | ✅ |
| **Prefix Cache** | ✅ (Radix) | ✅ | ✅ |
| **PD 分离** | ✅ (MemFabric) | ✅ | ✅ |
| **多节点** | ✅ | ✅ | ✅ |

### 10.4 适用场景

| 场景 | 推荐框架 |
|------|---------|
| 快速原型 / 研究 | SGLang |
| 生产部署 / 多硬件 | vLLM-Ascend |
| 企业级 / 深度优化 | MindIE-LLM |
| 昇腾硬件首发支持 | MindIE-LLM |
| 灵活定制 / 二次开发 | SGLang |

### 10.5 昇腾 NPU 硬件支持

| 硬件 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|------|------------|-------------|------------|
| Atlas 800I A2 | ✅ | ✅ | ✅ |
| Atlas 800I A3 | ✅ | ✅ | ✅ |
| Atlas A2 Training | ✅ | ✅ | ✅ |
| Atlas A3 Training | ✅ | ✅ | ✅ |
| Atlas 300I Duo | 实验性 | 待查 | 待查 |

### 10.6 版本依赖

| 依赖 | SGLang NPU | vLLM-Ascend |
|------|------------|-------------|
| CANN | 8.5.0 | 8.5.1 |
| PyTorch | 2.8.0 | 2.9.0 |
| torch-npu | 2.8.0.post2 | 2.9.0 |
| Triton | triton-ascend 3.2.0 | triton-ascend |

---

## 附录 A：关键代码路径

### SGLang NPU 核心路径
```
python/sglang/srt/hardware_backend/npu/
├── attention/ascend_backend.py           # 主attention实现
├── graph_runner/npu_graph_runner.py      # 图执行
├── allocator_npu.py                      # 内存分配
├── memory_pool_npu.py                    # 内存池
└── quantization/                         # 量化
```

### vLLM-Ascend 核心路径
```
vllm_ascend/
├── attention/attention_v1.py              # 主attention
├── worker/model_runner_v1.py              # 模型运行器
├── compilation/acl_graph.py              # 图编译
├── ops/                                  # 算子
└── spec_decode/                          # 推测解码
```

### MindIE-LLM 核心路径
```
src/
├── engine/llm_engine.cpp                 # 引擎主逻辑
├── scheduler/scheduler.cpp               # 调度器
├── block_manager/                         # KV Cache管理
├── llm_manager/                           # Python/C++桥接
└── server/endpoint/                      # 服务接口
```

---

## 附录 B：参考文献

1. SGLang Documentation: https://docs.sglang.io/
2. SGLang NPU Support: https://github.com/sgl-project/sglang/tree/main/docs_new/docs/hardware-platforms/ascend-npus
3. vLLM Ascend: https://github.com/vllm-project/vllm-ascend
4. vLLM Ascend Documentation: https://docs.vllm.ai/projects/ascend/en/latest/
5. MindIE-LLM: https://gitee.com/ascend/MindIE-LLM

---

*报告生成时间：2026-04-29*
*基于源码版本：SGLang (latest), vLLM-Ascend (v0.13.0), MindIE-LLM (2.3.0)*
