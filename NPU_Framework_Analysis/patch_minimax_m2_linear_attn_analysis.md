# patch_minimax_m2_linear_attn.py 分析

> 文件：`vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2_linear_attn.py`
> 分析日期：2026-05-19

## 1. 总览

该 patch 对 `MiniMaxText01RMSNormTP`（MiniMax-M2 线性注意力中的 TP RMSNorm 层，继承自 `CustomOp`）进行 3 处 monkey-patch：

| # | Patch 目标 | 目的 |
|---|-----------|------|
| 1 | `MiniMaxText01RMSNormTP.__init__` | 支持 `weight_shard_world_size` / `weight_shard_rank` 参数化分片 |
| 2 | `MiniMaxText01RMSNormTP.weight_loader` | 配套的分片权重加载器 |
| 3 | `MiniMaxText01RMSNormTP.forward_qk` | NPU 上用 `npu_rms_norm` 算子 + TP 全局方差修正 |

---

## 2. 背景：MiniMaxText01RMSNormTP 是什么

`MiniMaxText01RMSNormTP` 是 MiniMax-M2 模型中 Q/K 的 RMSNorm 层，**关键特点**：

- 继承 `CustomOp`（vLLM 的自定义算子调度基类）
- 支持 Tensor Parallel：权重在 TP 维度上分片
- Q 和 K 各有一份独立的 norm 参数（`q_norm`、`k_norm`）
- `forward_qk` 在同一函数内对 Q 和 K 同时做 RMSNorm，需要**跨 TP rank 求全局方差**

**文件：** `vllm/model_executor/layers/mamba/linear_attn.py:38-121`

**调用链：**
```
MiniMaxM2Attention.forward
    │
    └─ MiniMaxText01RMSNormTP.forward_qk(self.q_norm, self.k_norm, q, k)
            │
            ├─ 计算 q_var = q².mean(dim=-1)
            ├─ 计算 k_var = k².mean(dim=-1)
            ├─ TP > 1: all_reduce(q_var, k_var) / tp_world   ← 全局方差
            └─ q = q * rsqrt(q_var+eps) * q_norm.weight
               k = k * rsqrt(k_var+eps) * k_norm.weight
```

---

## 3. Patch 1 & 2: `__init__` + `weight_loader` — 参数化分片

### 原始实现（`linear_attn.py:40-78`）

```python
class MiniMaxText01RMSNormTP(CustomOp):
    def __init__(self, hidden_size, eps=1e-6, *, weight_shard_world_size=None, weight_shard_rank=None):
        super().__init__()                          # ← 调 CustomOp.__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight_shard_world = weight_shard_world_size or self.tp_world
        self.weight_shard_rank = self.tp_rank if weight_shard_rank is None else weight_shard_rank
        self.weight = nn.Parameter(torch.ones(hidden_size // self.weight_shard_world))
        self.weight.weight_loader = partial(self.weight_loader, ...)
        self.variance_epsilon = eps
```

原始 `__init__` 已经支持 `weight_shard_world_size` / `weight_shard_rank` 参数（用于 KV head 数小于 TP 时的复制分片场景）。

### Patch 改动

```python
class _patched_init(self, hidden_size, eps=1e-6, *, weight_shard_world_size=None, weight_shard_rank=None):
    CustomOp.__init__(self)                         # ← 直接调 CustomOp.__init__(), 不调 super()
    self.tp_world = get_tensor_model_parallel_world_size()
    self.tp_rank = get_tensor_model_parallel_rank()
    self.weight_shard_world = weight_shard_world_size or self.tp_world
    self.weight_shard_rank = self.tp_rank if weight_shard_rank is None else weight_shard_rank

    if hidden_size % self.weight_shard_world != 0:
        raise ValueError(...)

    self.weight = nn.Parameter(torch.ones(int(hidden_size / self.weight_shard_world)))
    self.weight.weight_loader = partial(_patched_weight_loader, ...)
    self.variance_epsilon = eps
```

**区别：**
- 原始调 `super().__init__()` → `CustomOp.__init__(enforce_enable=False, compile_native=False)`
- Patch 调 `CustomOp.__init__(self)` — 等价效果，但更显式处理不同上游版本的 `CustomOp` 签名差异
- 增加 `hidden_size % weight_shard_world != 0` 校验
- `weight_loader` 指向 patch 提供的 `_patched_weight_loader`

### `_patched_weight_loader`

```python
@staticmethod
def _patched_weight_loader(param, loaded_weight, shard_world_size=None, shard_rank=None):
    shard_size = loaded_weight.shape[0] // shard_world_size
    shard = slice(shard_rank * shard_size, (shard_rank + 1) * shard_size)
    param.data.copy_(loaded_weight[shard])
```

逻辑与原始一致——沿 dim=0 做 TP 分片拷贝。Patch 版本独立出来主要是为了绕过不同上游版本 `weight_loader` 签名的兼容性。

---

## 4. Patch 3: `forward_qk` — NPU fast path + TP 方差修正

### 原始实现（`linear_attn.py:101-121`）

```python
@staticmethod
def forward_qk(q_norm, k_norm, q, k):
    # Step 1: 局部方差
    q_var = q.to(float32).pow(2).mean(dim=-1, keepdim=True)
    k_var = k.to(float32).pow(2).mean(dim=-1, keepdim=True)

    # Step 2: TP 全局方差 (all_reduce)
    if q_norm.tp_world > 1:
        qk_var = torch.cat([q_var, k_var], dim=-1)
        qk_var = tensor_model_parallel_all_reduce(qk_var) / q_norm.tp_world
        q_var, k_var = qk_var.chunk(2, dim=-1)

    # Step 3: RMSNorm
    q = q * torch.rsqrt(q_var + eps) * q_norm.weight
    k = k * torch.rsqrt(k_var + eps) * k_norm.weight
    return q, k
```

**数据处理流程（原始）：**
```
就地计算 q_var, k_var
    ↓
[TP all_reduce] → qk_var  (合并所有 rank 的方差)
    ↓
q = q * rsqrt(q_var + eps) * weight
k = k * rsqrt(k_var + eps) * weight
    ↓
return q, k
```

### Patch 实现

```python
@staticmethod
def _patched_qk(q_norm, k_norm, q, k):
    if current_platform.device_name == "npu":
        # ---- NPU fast path ----
        # Step 1: 用 npu_rms_norm 算子做局部 RMSNorm
        q, q_inv_rms = torch.ops.npu.npu_rms_norm(q, q_norm.weight, eps)
        k, k_inv_rms = torch.ops.npu.npu_rms_norm(k, k_norm.weight, eps)

        if q_norm.tp_world > 1:
            # Step 2: 从 inv_rms 反推局部方差, all_reduce 得全局方差
            q_local_var = (q_inv_rms².reciprocal() - eps).clamp_min(0)
            k_local_var = (k_inv_rms².reciprocal() - eps).clamp_min(0)
            qk_var = all_reduce(cat(q_local_var, k_local_var)) / tp_world
            q_global_var, k_global_var = qk_var.chunk(2)

            # Step 3: 用全局方差/局部方差的比值修正 norm 输出
            q = q * (rsqrt(q_global_var + eps) / rsqrt(q_local_var + eps))
            k = k * (rsqrt(k_global_var + eps) / rsqrt(k_local_var + eps))

        return q, k

    # 非 NPU: 走原始实现
    return _original_qk_method(q_norm, k_norm, q, k)
```

**NPU fast path 公式推导：**

`npu_rms_norm` 算子输出 `(output, inv_rms)`，其中：

$$
\text{output} = x \cdot \text{rsqrt}(x^2.\text{mean}(-1) + \epsilon) \cdot w
$$

$$
\text{inv\_rms} = \text{rsqrt}(x^2.\text{mean}(-1) + \epsilon)
$$

从 `inv_rms` 反推局部方差：

$$
\sigma^2_{\text{local}} = \left(\frac{1}{\text{inv\_rms}}\right)^2 - \epsilon = x^2.\text{mean}(-1)
$$

TP 全局方差：

$$
\sigma^2_{\text{global}} = \frac{\sum_{\text{ranks}} \sigma^2_{\text{local}}}{\text{tp\_world}}
$$

修正 output（让 norm 结果等价于用全局方差）：

$$
x_{\text{final}} = x_{\text{npu\_out}} \cdot \frac{\text{rsqrt}(\sigma^2_{\text{global}} + \epsilon)}{\text{rsqrt}(\sigma^2_{\text{local}} + \epsilon)}
$$

因为 `npu_rms_norm` 输出已经乘了 `rsqrt(σ²_local + ε) * w`，所以只需要把 `rsqrt(σ²_local + ε)` 替换为 `rsqrt(σ²_global + ε)`。

### 对比：原始 vs NPU fast path

| 步骤 | 原始实现 | NPU fast path |
|------|---------|---------------|
| 局部 norm | `q * rsqrt(q².mean() + eps) * weight` (Python) | `npu_rms_norm(q, weight, eps)` (AscendC kernel) |
| 方差计算 | `q².mean(dim=-1)` (Python float32) | 从 `inv_rms` 反推 (更少的内存读写) |
| all_reduce | `q_var, k_var` concat 后 all_reduce | `q_local_var, k_local_var` concat 后 all_reduce (相同) |
| 全局纠正 | 直接在 Python 中 `q * rsqrt(global_var) * weight` | `q * rsqrt(global_var) / rsqrt(local_var)` 修正已有输出 |

核心优势：NPU fast path 用 `npu_rms_norm` 融合算子替代 Python 中的多次 tensor 运算，减少 host-device 交互和中间结果 materialize。

### 时序（NPU 路径）

```
┌──────────┐     ┌─────────────────────┐     ┌──────────┐     ┌──────────────┐
│ Attn.    │     │ _patched_qk         │     │ npu      │     │ all_reduce   │
│ forward  │     │ (MiniMaxRMSNormTP)  │     │ rms_norm │     │ (TP 通信)     │
└────┬─────┘     └──────────┬──────────┘     └────┬─────┘     └──────┬───────┘
     │                      │                    │                  │
     │ forward_qk(q,k)      │                    │                  │
     │─────────────────────▶│                    │                  │
     │                      │                    │                  │
     │                      │ npu_rms_norm(q)    │                  │
     │                      │───────────────────▶│                  │
     │                      │←── q, q_inv_rms ───│                  │
     │                      │                    │                  │
     │                      │ npu_rms_norm(k)    │                  │
     │                      │───────────────────▶│                  │
     │                      │←── k, k_inv_rms ───│                  │
     │                      │                    │                  │
     │                      │ tp_world > 1?      │                  │
     │                      │── YES ──────────────────────────────────────────
     │                      │                    │                  │
     │                      │ 反推 q_local_var = (1/inv_rms)² - eps │
     │                      │ 反推 k_local_var = (1/inv_rms)² - eps │
     │                      │                    │                  │
     │                      │ all_reduce(qk_var) │                  │
     │                      │──────────────────────────────────────▶│
     │                      │←── global_qk_var ─────────────────────│
     │                      │                    │                  │
     │                      │ q *= rsqrt(global_var) / rsqrt(local_var)
     │                      │ k *= rsqrt(global_var) / rsqrt(local_var)
     │                      │                    │                  │
     │                      │ return (q, k)      │                  │
     │◀─────────────────────│                    │                  │
```

---

## 5. 与 `patch_minimax_m2.py` 的关系

`patch_minimax_m2.py` 中 `MiniMaxM2Attention` 使用了两处来自本 patch 的能力：

```
MiniMaxM2Attention.__init__ (patch_minimax_m2.py)
    │
    ├─ 当 total_num_kv_heads < tp_size:
    │     k_norm = MiniMaxText01RMSNormTP(
    │         head_dim * total_num_kv_heads,
    │         weight_shard_world_size=total_num_kv_heads,   ← Patch 1 (本文件)
    │         weight_shard_rank=tp_rank // num_kv_replicas,  ← Patch 1 (本文件)
    │     )

MiniMaxM2Attention.forward (patch_minimax_m2.py)
    │
    └─ split_qkv_tp_rmsnorm_rope(...)
         (融合算子内置了 RMSNorm, 不走 MiniMaxText01RMSNormTP.forward_qk)
         (但 q_norm.weight / k_norm.weight 仍通过 Patch 2 的 weight_loader 加载)
```

## 6. 兼容性处理

```python
# 处理不同 vLLM 版本的 forward_qk 方法名差异
if hasattr(MiniMaxText01RMSNormTP, "forward_qk"):
    _ORIG_QK_METHOD_NAME = "forward_qk"
elif hasattr(MiniMaxText01RMSNormTP, "_normalize_qk"):
    _ORIG_QK_METHOD_NAME = "_normalize_qk"   # 旧版 vLLM

# 检测原始方法是否为 staticmethod (不同版本可能不同)
_qk_is_staticmethod = isinstance(MiniMaxText01RMSNormTP.__dict__.get(_ORIG_QK_METHOD_NAME), staticmethod)

# 始终以 staticmethod 方式安装 patch
setattr(MiniMaxText01RMSNormTP, _ORIG_QK_METHOD_NAME, staticmethod(_patched_qk))
```
