# SGLang vs MindIE-LLM vs vLLM-Ascend 详细逐项对比分析

## 目录

1. [Attention 实现详细对比](#1-attention-实现详细对比)
2. [内存管理和 KV Cache 对比](#2-内存管理和-kv-cache-对比)
3. [调度器实现对比](#3-调度器实现对比)
4. [分布式通信对比](#4-分布式通信对比)
5. [量化和压缩技术对比](#5-量化和压缩技术对比)
6. [推测解码实现对比](#6-推测解码实现对比)
7. [模型适配和扩展性对比](#7-模型适配和扩展性对比)

---

## 1. Attention 实现详细对比

### 1.1 Attention 核心接口

#### SGLang NPU

```python
# 文件：python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py

class AscendAttnBackend(AttentionBackend):
    """SGLang NPU Attention 后端"""

    def __init__(self, model_runner: ModelRunner, speculative_step_id: int = 0):
        # 初始化配置
        self.forward_metadata = None
        self.device = model_runner.device
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.graph_mode = False

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化前向元数据"""
        # 处理 block_tables, seq_lens, prefix_lens 等
        pass

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache, ...):
        """Prefill 阶段 Attention"""
        pass

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache, ...):
        """Decode 阶段 Attention"""
        pass

    def forward_mtp(self, q, k, v, layer, forward_batch, save_kv_cache, ...):
        """多步推测 Attention"""
        pass
```

**关键设计**：
1. **Forward 模式分离**：通过方法覆盖实现不同阶段的高效处理
2. **MLA 优先检测**：`self.use_mla` 决定使用哪种 Attention 路径
3. **FIA/Torch 原生切换**：通过 `ASCEND_USE_FIA` 环境变量控制

#### vLLM-Ascend

```python
# 文件：vllm_ascend/attention/attention_v1.py

class AscendAttentionBackend(AttentionBackend):
    """vLLM-Ascend Attention 后端（插件化）"""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            return AscendAttentionCPImpl  # Context Parallel
        return AscendAttentionBackendImpl

class AscendAttentionBackendImpl(AttentionImpl):
    """实际 Attention 实现"""

    def forward(self, layer, query, key, value, kv_cache,
                attn_metadata, output, ...):
        """统一 forward 接口"""
        num_tokens = query.shape[0]
        if using_paged_attention(...):
            return self.forward_paged_attention(query, attn_metadata, output)
        else:
            return self.forward_fused_infer_attention(
                query, key, value, attn_metadata, output, kv_cache
            )
```

**关键设计**：
1. **状态机管理**：`AscendAttentionState` 枚举管理不同状态
2. **插件化注册**：通过 `@register_backend` 装饰器注册
3. **FIA/PA 双路径**：根据条件选择 fused attention 或 paged attention

#### MindIE-LLM

```cpp
// 文件：src/engine/construct_execute_request.cpp

// C++ 实现，无直接 Python Attention 接口
// 通过 C++ 核心处理 Attention 计算

class LlmEngine {
    void ScheduleExecTransfer(std::shared_ptr<EnginePerDP>& engine) const {
        // 调度执行和传输
    }

    SchOutDataPair PostScheduleSyncUp(...) {
        // 同步和调度后处理
    }
};
```

**关键设计**：
1. **全 C++ 实现**：性能关键路径完全在 C++
2. **调度器集成**：Attention 与 Scheduler 深度耦合
3. **无 Python Attention 接口**：通过底层算子封装

---

### 1.2 Attention Kernel 调用对比

#### SGLang NPU - FIA Kernel

```python
# Prefill 阶段使用 FIA
torch.ops.npu.npu_fused_infer_attention_score(
    query=q,                           # [bs, num_heads, head_dim] 或 [token, num_heads, head_dim]
    key=k,                             # K cache
    value=v,                           # V cache
    num_heads=layer.tp_q_head_num,
    num_key_value_heads=layer.tp_k_head_num,
    input_layout="BSND",  # 或 "TND"
    atten_mask=self.fia_mask,
    sparse_mode=3,         # 3=因果注意力
    scale=layer.scaling,
    actual_seq_lengths=seq_lens,
    actual_seq_lengths_kv=seq_lens,
)
```

#### SGLang NPU - Paged Attention

```python
# Decode 阶段使用 Paged Attention
torch_npu._npu_paged_attention(
    query=query.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
    key_cache=k_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache=v_cache,
    num_heads=layer.tp_q_head_num,
    num_kv_heads=layer.tp_k_head_num,
    scale_value=layer.scaling,
    block_table=block_tables, # [batch_size, max_blocks_per_seq]
    context_lens=seq_lens,
    out=attn_output,
)
```

#### vLLM-Ascend - C8 量化 Attention

```python
# INT8 KV Cache 的 Attention 实现
torch_npu.npu_fused_infer_attention_score(
    query=query.unsqueeze(2),  # [batch, 1, num_heads, head_dim]
    key=key_cache,
    value=value_cache,
    key_antiquant_scale=layer._c8_k_aq_scale,
    key_antiquant_offset=layer._c8_k_aq_offset,
    value_antiquant_scale=layer._c8_v_aq_scale,
    value_antiquant_offset=layer._c8_v_aq_offset,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
    input_layout="BNSD",
    sparse_mode=0,
)
```

---

### 1.3 MLA (Multi-head Latent Attention) 实现

#### SGLang NPU - Ring MLA

```python
# MLA 预处理器
torch_npu.atb.npu_ring_mla(
    q_nope=q_nope,          # [token, kv_heads, kv_lora_rank]
    q_rope=q_rope,          # [token, kv_heads, rope_head_dim]
    k_nope=k_nope,
    k_rope=k_rope,
    value=v,
    mask=self.ringmla_mask,
    seqlen=seq_lens,
    head_num=layer.tp_q_head_num,
    kv_head_num=layer.tp_k_head_num,
    pre_out=None,           # 首次调用
    prev_lse=None,          # 首次调用
    qk_scale=layer.scaling,
    kernel_type="kernel_type_high_precision",
    mask_type="mask_type_triu",  # 上三角 mask
    calc_type="calc_type_first_ring",
    output=attn_output,
    softmax_lse=attn_lse,
)

# 第二次调用（合并历史）
torch_npu.atb.npu_ring_mla(
    pre_out=attn_output,     # 传入之前的输出
    prev_lse=attn_lse,       # 传入之前的 LSE
    calc_type="calc_type_default",  # 增量计算
    mask_type="no_mask",
)
```

#### vLLM-Ascend - MLA via FIA

```python
# MLA 通过 npu_fused_infer_attention_score 实现
# TND layout 支持 MLA 的特殊格式

attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
    q_nope,                  # [token, num_heads, kv_lora_rank]
    k_nope,                  # 压缩的 K
    v,                       # V 值
    query_rope=q_rope,
    key_rope=k_rope,
    num_heads=layer.tp_q_head_num,
    input_layout="TND",
    atten_mask=self.fia_mask,
    sparse_mode=3,
    actual_seq_lengths=seq_lens_list_cumsum,
    actual_seq_lengths_kv=seq_lens_list_cumsum,
    scale=layer.scaling,
)
```

---

## 2. 内存管理和 KV Cache 对比

### 2.1 分页分配器实现

#### SGLang NPU - NPUPagedTokenToKVPoolAllocator

```python
# 文件：python/sglang/srt/hardware_backend/npu/allocator_npu.py

class NPUPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """NPU 专用分页分配器"""

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens,
                     seq_lens_cpu, last_loc, extend_num_tokens):
        """分配扩展请求的 KV Cache"""

        # 计算需要的新页数
        num_new_pages = (
            (seq_lens + self.roundup) // self.page_size
            - (prefix_lens + self.roundup) // self.page_size
        ).sum()

        # 小批量使用优化 kernel
        if num_new_pages_item < 200:
            from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel
            alloc_extend_kernel[(bs,)](...)  # GPU kernel
        else:
            # 大批量使用朴素算法
            alloc_extend_naive(...)

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        """分配解码请求的 KV Cache"""
        num_new_pages = get_num_new_pages(seq_lens_cpu, self.page_size, decode=True)

        # 特殊情况处理
        need_new_pages = (seq_lens % self.page_size == 1).int()
        # ...

    def free(self, free_index):
        """释放 KV Cache"""
        # 支持排序优化
        if self.need_sort:
            self.release_pages = torch.cat((free_page_indices, self.release_pages))
        else:
            self.free_pages = torch.cat((free_page_indices, self.free_pages))
```

**设计亮点**：
1. **小批量优化**：<200 页使用专用 kernel
2. **排序释放**：优化内存碎片
3. **调试模式**：可选完整性检查

#### vLLM-Ascend - Block 管理

```python
# vLLM 标准 PagedAttention 机制
# 通过 slot_mapping 和 block_table 映射

class AscendAttentionMetadataBuilder:
    def build(self, common_prefix_len, common_attn_metadata):
        # 构建元数据
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        return AscendMetadata(
            block_tables=block_table,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            ...
        )
```

#### MindIE-LLM - C++ Block Manager

```cpp
// 文件：src/block_manager/self_attn_block_manager.cpp

class SelfAttnBlockManager {
    BlockSpaceManagerSPtr blockManager_;

    // 支持的功能：
    // 1. LRU 驱逐
    // 2. Prefix Cache
    // 3. Copy-on-Write (CoW)
    // 4. Swap (CPU-NPU)

    Status Alloc(const std::vector<TokenId>& seq,
                std::vector<BlockId>* allocated_blocks) {
        // 分配块
    }

    Status Free(const std::vector<BlockId>& blocks) {
        // 释放块
    }
};
```

---

### 2.2 Prefix Cache 实现

#### SGLang NPU - RadixAttention

```python
# SGLang 使用 RadixAttention 实现高效的 Prefix Cache

class RadixAttention:
    """基数树注意力，实现 KV Cache 共享"""

    def match_prefix(self, req_to_token, token_ids):
        """匹配前缀，返回缓存命中信息"""
        # 在基数树中查找最长前缀匹配
        pass

    def cache_attention(self, req_pool_indices, ...):
        """缓存注意力结果"""
        pass

# Attention 后端中使用
if sum(forward_batch.extend_prefix_lens_cpu) > 0:
    # 使用 prefix cache
    self.forward_metadata.prefix_lens = forward_batch.extend_prefix_lens
    # 索引前缀块
    flatten_prefix_block_tables = torch.index_select(
        k_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
    )
```

#### vLLM-Ascend - Hash-based Prefix Cache

```python
# vLLM 使用 hash 方法实现 prefix cache

# vllm_ascend/patch/worker/patch_prefix_cache.py
# 通过 hash 匹配前缀
```

---

## 3. 调度器实现对比

### 3.1 SGLang - ForwardMode 调度

```python
# python/sglang/srt/layers/radix_attention.py

class ForwardMode(Enum):
    EXTEND = "extend"           # Prefill
    DECODE = "decode"           # Decode
    EXTEND_V1 = "extend_v1"
    DRAFT_EXTEND = "draft_extend"    # 推测扩展
    TARGET_VERIFY = "target_verify"   # 目标验证
    IDLE = "idle"

# 调度流程
class Scheduler:
    def schedule(self):
        # 1. 获取待调度请求
        # 2. 根据 ForwardMode 分发
        # 3. 执行 attention
        pass
```

**特点**：
- 简单直接的状态机
- 与 RadixAttention 紧耦合
- 适合单一模型场景

### 3.2 vLLM-Ascend - v1 Scheduler

```python
# vllm/v1/core/sched/scheduler.py

class Scheduler:
    """vLLM v1 调度器"""

    def schedule(self):
        # 1. 分割 decode 和 prefill
        num_decodes, num_prefills = split_decodes_and_prefills(...)

        # 2. 应用 chunked prefill
        if self.enable_chunked_prefill:
            # 分块处理长序列

        # 3. 调度预算
        budget = SchedulingBudget(...)
        for seq in decode_batch:
            budget.consume_tokens(num_tokens)

        # 4. 输出 SchedulerOutput
        return SchedulerOutput(...)
```

**特点**：
- Continuous Batching
- Chunked Prefill
- 与 vLLM 架构深度集成

### 3.3 MindIE-LLM - 多策略调度

```cpp
// src/scheduler/scheduler.h

class Scheduler : public IScheduler {
    // 调度策略
    std::shared_ptr<Policy> prefillPolicy_;   // 预填充策略
    std::shared_ptr<Policy> decodePolicy_;    // 解码策略
    std::shared_ptr<StagePolicy> stagePolicy_;
    std::shared_ptr<DynamicBatchSize> dynamicBatchSize_;

    // 核心调度方法
    std::pair<SequenceGroupMetaDatas, SchedulerOutputs>
    Schedule(bool needSync = false) override {
        // 1. 入队新请求
        // 2. 选择调度策略
        // 3. 分配 KV Cache
        // 4. 输出调度结果
    }

    // 预填充和解码分离调度
    std::pair<SequenceGroupMetaDatas, SchedulerOutputs>
    ScheduleTransfer() override {
        // P-D 分离场景的传输调度
    }
};

// 策略模式
class Policy {
    virtual std::vector<SequenceGroupSPtr>
    select(std::deque<SequenceGroupSPtr>& candidates,
           SchedulingBudget& budget) = 0;
};

class FCFSPolicy : public Policy {
    // 先来先服务
};

class PDDSPolicy : public Policy {
    // 预定义调度策略
};
```

**特点**：
- 策略模式支持多种调度算法
- 支持动态批大小
- 支持 Layerwise 调度
- 多 DP 并行支持

---

## 4. 分布式通信对比

### 4.1 NPU 通信器

#### SGLang NPU

```python
# python/sglang/srt/distributed/device_communicators/npu_communicator.py

class NPUCommunicator:
    """NPU 通信器，基于 HCCL """

    def __init__(self, tp_size, tp_rank):
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def all_reduce(self, input_):
        """全局归约"""
        # 调用 HCCL all_reduce
        torch.distributed.all_reduce(
            input_,
            op=torch.distributed.ReduceOp.SUM,
            group=self.tp_group
        )

    def broadcast(self, input_, src=0):
        """广播"""
        torch.distributed.broadcast(input_, src, group=self.tp_group)

    def all_gather(self, input_):
        """全局收集"""
        torch.distributed.all_gather_object(input_, ...)
```

#### vLLM-Ascend

```python
# vllm_ascend/distributed/device_communicators/npu_communicator.py

class NPUCommunicator:
    """vLLM-Ascend NPU 通信器"""

    def __init__(self, device_group):
        self.device_group = device_group

    def all_reduce(self, tensor, op=ReduceOp.SUM):
        # PyHCCL 实现
        from vllm_ascend.distributed.device_communicators.pyhccl import pyhccl

        pyhccl.all_reduce(tensor, self.device_group, op)

    def broadcast(self, tensor, root=0):
        pyhccl.broadcast(tensor, self.device_group, root)
```

#### MindIE-LLM

```cpp
// src/executor/communicator.h

class Communicator {
    std::shared_ptr<ProcessGroup> process_group_;

public:
    void AllReduce(Tensor& tensor, const ReduceOp& op = ReduceOp::SUM) {
        process_group_->AllReduce(tensor, op);
    }

    void Broadcast(Tensor& tensor, int root_rank) {
        process_group_->Broadcast(tensor, root_rank);
    }

    void AllGather(Tensor& tensor) {
        process_group_->AllGather(tensor);
    }
};

// 初始化
void LlmEngine::InitProcessGroup(
    const std::vector<NodeInfo>& nodeInfos,
    std::string& masterIP,
    uint32_t masterPort
) {
    process_group_ = std::make_shared<ProcessGroup>(nodeInfos, masterIP, masterPort);
}
```

---

### 4.2 Expert Parallelism 对比

#### SGLang NPU - DeepEP 兼容

```python
# 使用 sgl-kernel-npu 中的 DeepEP 替代库
# python/deep_ep/README.md

# 初始化
from deep_ep import get_expert_parallel_group

ep_group = get_expert_parallel_group()
tp_group = get_tensor_model_parallel_group()

# 发送
def send_tensor(tensor, dst, ep_group):
    # 跨 EP rank 发送
    pass

# 接收
def recv_tensor(tensor, src, ep_group):
    pass
```

#### vLLM-Ascend - EPLB

```python
# vllm_ascend/eplb/

class VllmEplbAdaptor:
    """vLLM EPLB 适配器"""

    def load_expert_weights(self, expert_weights):
        # 加载专家权重
        pass

    def all_reduce_expert_output(self, output):
        # 跨专家归约
        pass

# 使用
class EplbWorker:
    def execute_model(self, input_batch):
        # EPLB 执行
        expert_output = self.eplb.forward(input_batch)
        return expert_output
```

---

## 5. 量化和压缩技术对比

### 5.1 W8A8 量化

#### SGLang NPU

```python
# python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py

class W8A8DynamicLinearMethod:
    """W8A8 动态量化"""

    def process_weights_after_loading(self, layer):
        # 获取权重
        weight = layer.weight.data  # [out_features, in_features]

        # 动态量化
        scales = weight.abs().max(dim=-1).values / 127.0
        quantized_weight = (weight / scales.to(weight.dtype)).to(torch.int8)

        self.scales = scales
        self.quantized_weight = quantized_weight

    def forward(self, input_):
        # INT8 Matmul
        output = torch.matmul(input_, self.quantized_weight.t())

        # 反量化
        output = output * self.scales

        return output
```

#### vLLM-Ascend

```python
# vllm_ascend/quantization/methods/w8a8_dynamic.py

class W8A8DynamicMethod:
    """vLLM-Ascend W8A8 动态量化"""

    def create_weights(self, layer):
        # 权重量化
        self.weight = torch.nn.Parameter(
            weight.to(torch.float8_e4m3fnuz)
        )
        self.weight_scale = ...

    def apply(self, x, bias):
        # 使用 torch_npu 的量化 matmul
        return torch_npu.llm_api_fused_mm_dequantize(
            x, self.weight, self.weight_scale
        )
```

### 5.2 C8 (INT8 KV Cache) 量化

#### vLLM-Ascend 独有实现

```python
# vllm_ascend/quantization/methods/kv_c8.py

class KVCacheC8QuantMethod:
    """INT8 KV Cache 量化 (C8)"""

    def create_weights(self, layer):
        # KV Cache 的 per-channel 量化
        self.k_cache_scale = torch.nn.Parameter(
            torch.ones(num_kv_heads, head_size)
        )
        self.k_cache_offset = torch.nn.Parameter(
            torch.zeros(num_kv_heads, head_size)
        )

    def process_after_compute(self, k, v, layer):
        # 量化 K, V 到 INT8
        k_int8 = torch.clamp(
            torch.round(k * inv_scale + offset),
            -128, 127
        ).to(torch.int8)

        v_int8 = torch.clamp(
            torch.round(v * inv_scale + offset),
            -128, 127
        ).to(torch.int8)

        return k_int8, v_int8

# Attention 中使用
class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    def forward(self, query, key, value, kv_cache, attn_metadata, output, ...):
        # 量化
        k_int8, v_int8 = self._quantize_kv_to_int8(k, v, layer, ...)

        # 使用 antiquant 进行 attention
        output = torch_npu.npu_fused_infer_attention_score(
            query, k_int8, v_int8,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            ...
        )
```

---

## 6. 推测解码实现对比

### 6.1 SGLang NPU - Eagle/Eagle3

```python
# python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py

class EagleDraftNPUGraphRunner:
    """Eagle 推测解码的 Graph Runner"""

    def __init__(self, model, config):
        self.model = model
        self.num_draft_tokens = config.speculative_num_draft_tokens

        # 为每个 draft step 创建 attention backend
        self.attn_backends = []
        for step_id in range(self.num_draft_tokens):
            self.attn_backends.append(
                AscendAttnBackend(model_runner, speculative_step_id=step_id)
            )

    def draft_forward(self, forward_batch):
        """Draft 模型前向"""
        # 1. Draft token 生成
        # 2. Verify 阶段处理

        outputs = []
        for step_id in range(self.num_draft_tokens):
            output = self.model(
                input_ids=draft_tokens[step_id],
                forward_batch=forward_batch,
                spec_metadata=self.spec_metadata[step_id]
            )
            outputs.append(output)

        return outputs

    def verify_forward(self, forward_batch):
        """Verify 阶段前向"""
        # 使用 target model 验证 draft tokens
        pass
```

### 6.2 vLLM-Ascend - 推测解码

```python
# vllm_ascend/spec_decode/eagle_proposer.py

class AscendEagleProposer:
    """Eagle 推测解码实现"""

    def __init__(self, model_runner, config):
        self.model_runner = model_runner
        self.num_speculative_tokens = config.num_speculative_tokens

    def propose(self, forward_context):
        """生成推测 tokens"""
        # 1. 获取 draft 模型输出
        # 2. 生成 speculative tokens
        # 3. 返回提议

    def verify(self, output):
        """验证推测结果"""
        # 验证 accept/reject
        pass

# NPUGraph 推测解码支持
class AscendAttentionBackendImpl:
    def update_graph_params(self, update_stream, forward_context, ...):
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
            # 处理推测解码的图更新
```

### 6.3 MindIE-LLM - 推测解码

```cpp
// C++ 实现
// src/engine/llm_engine.h

class LlmEngine {
    // 支持推测解码
    // 与 Scheduler 深度集成

    std::pair<SequenceGroupMetaDatas, SchedulerOutputs>
    Schedule(...) override {
        // 1. 正常调度
        // 2. 推测解码调度
    }
};
```

---

## 7. 模型适配和扩展性对比

### 7.1 新模型适配

#### SGLang NPU

**适配步骤**：
1. 注册模型架构
2. 实现/继承 `RadixAttention`
3. 配置 `model_config`
4. 可选：自定义 Attention Backend

```python
# python/sglang/srt/models/ 注册新模型
@register_model("MyCustomModel")
class MyCustomForCausalLM(nn.Module):
    def __init__(self, config, ...):
        self.layers = nn.ModuleList([MyDecoderLayer() for _ in range(config.num_hidden_layers)])
        self.attn = RadixAttention(...)

    def forward(self, input_ids, ...):
        # 实现
        pass
```

#### vLLM-Ascend

**适配方式**：通过 Patch 机制

```python
# vllm_ascend/patch/worker/patch_custom_model.py

# 1. 创建 patch
def patch_custom_model():
    from vllm.model_executor.models.custom_model import CustomModel

    # 修改 forward
    def new_forward(self, ...):
        # 自定义实现
        pass

    CustomModel.forward = new_forward

# 2. 注册 patch
# 在 model_runner 初始化时应用
```

**设计优势**：
- 无需修改上游模型
- 通过猴子补丁实现定制
- 可选的长期方案：上游贡献

### 7.2 自定义算子注册

#### SGLang NPU

```python
# 通过 sgl-kernel-npu 注册

# python/sgl_kernel_npu/__init__.py
from sgl_kernel_npu import register_ops

@register_ops
class MyCustomOp:
    @staticmethod
    def forward(x, y):
        return torch_npu.custom_op(x, y)
```

#### vLLM-Ascend

```python
# vllm_ascend/ops/register_custom_ops.py

# 注册 Triton 自定义算子
from vllm_ascend.ops.triton import triton_ops

@triton_ops.register
class MyCustomTritonOp:
    @staticmethod
    def forward(x, y):
        # Triton kernel 实现
        pass
```

---

## 附录：关键文件索引

### SGLang NPU 核心文件

| 文件路径 | 功能描述 |
|---------|---------|
| `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | NPU Attention 后端主实现 |
| `python/sglang/srt/hardware_backend/npu/allocator_npu.py` | 分页内存分配器 |
| `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` | CUDA Graph 执行器 |
| `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` | NPU 内存池 |
| `python/sglang/srt/distributed/device_communicators/npu_communicator.py` | NPU 集合通信 |

### vLLM-Ascend 核心文件

| 文件路径 | 功能描述 |
|---------|---------|
| `vllm_ascend/attention/attention_v1.py` | Attention 后端实现 |
| `vllm_ascend/worker/model_runner_v1.py` | 模型运行器 |
| `vllm_ascend/compilation/acl_graph.py` | 图编译支持 |
| `vllm_ascend/distributed/device_communicators/npu_communicator.py` | NPU 通信 |
| `vllm_ascend/spec_decode/` | 推测解码实现 |

### MindIE-LLM 核心文件

| 文件路径 | 功能描述 |
|---------|---------|
| `src/engine/llm_engine.cpp` | LLM 引擎主逻辑 |
| `src/scheduler/scheduler.cpp` | 调度器实现 |
| `src/block_manager/self_attn_block_manager.cpp` | KV Cache 块管理 |
| `src/executor/communicator.cpp` | 集合通信 |
| `src/llm_manager/llm_manager.cpp` | Python/C++ 桥接 |

---

*详细对比分析完成*
