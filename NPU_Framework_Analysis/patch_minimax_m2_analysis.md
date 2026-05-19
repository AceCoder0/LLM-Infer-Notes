# patch_minimax_m2.py 分析

> 文件：`vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py`
> 分析日期：2026-05-19

## 1. 总览

该 patch 对 MiniMax-M2 模型进行 4 处 monkey-patch，全部在 `import` 时自动执行：

| # | Patch 目标 | 目的 |
|---|-----------|------|
| 1 | `MiniMaxM2MoE.forward` | TP 模式下 MoE 输出走 `maybe_all_reduce_tensor_model_parallel` |
| 2 | `MiniMaxM2Attention.__init__` / `.forward` | KV head 不足 TP 时增加 k_norm 分片 + 融合 QKV RMSNorm RoPE 算子 |
| 3 | `MiniMaxM2Model.load_weights` | FP8 checkpoint 反量化到 BF16 加载 |
| 4 | `MiniMaxM2Model.forward` / `MiniMaxM2ForCausalLM` | Eagle3 投机解码 aux hidden states 支持 |

---

## 2. Patch 1: MiniMaxM2MoE.forward — TP all_reduce

### 被 Patch 的函数

**文件：** `vllm/model_executor/models/minimax_m2.py:130-140`

```python
def forward(self, hidden_states):
    router_logits, _ = self.gate(hidden_states.to(torch.float32))
    final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
    return final_hidden_states.view(num_tokens, hidden_dim)
```

原始实现直接返回 MoE 输出，不显式调用 all_reduce。

### Patch 内容

```python
def _patched_moe_forward(self, hidden_states):
    # ... 同原始逻辑 ...
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)
    return final_hidden_states.view(num_tokens, hidden_dim)
```

增加一行 `maybe_all_reduce_tensor_model_parallel`，确保在 TP 模式下各 rank 的 MoE 输出正确合并。

### 时序

```
MiniMaxM2MoE.forward
    │
    ├─ gate(hidden_states)           ← router 计算
    ├─ experts(hidden_states, ...)   ← MoE 实际计算
    │
    ├─ tp_size > 1 ?
    │   └─ YES → maybe_all_reduce_tensor_model_parallel
    │            (各 TP rank 的 expert 输出做 all_reduce)
    │
    └─ return final_hidden_states
```

---

## 3. Patch 2: MiniMaxM2Attention — k_norm 分片 + 融合算子

### 3a. `__init__` 补丁：KV head 不足 TP 时的 k_norm 分片

**原始逻辑**（`minimax_m2.py:144-238`）：

```python
# 原始: total_num_kv_heads >= tp_size 时正常创建 k_norm
if self.total_num_kv_heads >= tp_size:
    self.k_norm = MiniMaxText01RMSNormTP(head_dim * total_num_kv_heads, eps=rms_norm_eps)
else:
    # 原始也处理了 < tp_size 的情况，但用的是默认 weight_shard_rank
    num_kv_head_replicas = tp_size // self.total_num_kv_heads
    self.k_norm = MiniMaxText01RMSNormTP(
        head_dim * self.total_num_kv_heads, eps=rms_norm_eps,
        weight_shard_world_size=self.total_num_kv_heads,
        weight_shard_rank=get_tensor_model_parallel_rank() // num_kv_head_replicas,
    )
```

**Patch 增加**：`self.num_kv_head_replicas = max(1, tp_size // self.total_num_kv_heads)` 并额外覆盖 `total_num_kv_heads < tp_size` 的情况。原始的 `__init__` 其实已经处理了这种情况 — 但 Patch 增加 `num_kv_head_replicas` 属性作为后续 forward 使用的元数据。

### 3b. `forward` 补丁：融合 QKV split + RMSNorm + RoPE

**原始 forward**（`minimax_m2.py:240-251`）：

```python
def forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = MiniMaxText01RMSNormTP.forward_qk(self.q_norm, self.k_norm, q, k)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output
```

**Patch 后**：用 `torch.ops.vllm.split_qkv_tp_rmsnorm_rope` 融合算子一步完成 split + norm + rope：

```python
def _patch_forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    cos, sin = get_cos_and_sin_slice()
    q, k, v = torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
        input=qkv, q_weight=self.q_norm.weight, k_weight=self.k_norm.weight,
        q_hidden_size=self.q_size, kv_hidden_size=self.kv_size,
        head_dim=self.head_dim, rotary_dim=..., eps=...,
        tp_world=self.q_norm.tp_world, cos=cos, sin=sin,
    )
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output
```

**对比：**

```
原始流程:
  qkv_proj → split(3段) → forward_qk(RMSNorm) → RoPE → attn → o_proj
                    ↑ 3 次 kernel launch

Patch 流程:
  qkv_proj → split_qkv_tp_rmsnorm_rope (1 次融合 kernel) → attn → o_proj
                    ↑ 减少 kernel launch 开销
```

---

## 4. Patch 3: MiniMaxM2Model.load_weights — FP8 反量化加载

### 被 Patch 的函数

`MiniMaxM2Model.load_weights`（`minimax_m2.py:420-516`）是手动编写的 weight loader，处理 QKV stacked mapping、MoE expert mapping、KV scale remap 等。

### 问题

MiniMax-M2 的 FP8 checkpoint 把权重存为 `float8_e4m3fn`，附带 `weight_scale_inv`（per-block 反量化 scale）。NPU 上直接加载 FP8 权重有兼容性问题，需要反量化到 BF16。

### Patch 架构

```
_patched_load_weights(self, weights)
    │
    ├─ _need_dequantize_fp8_weights()?
    │   ├─ NO  → 直接调 original_load_weights
    │   └─ YES → _fp8_dequant_weight_iter(weights)
    │            ↓
    │            遍历 weights 流:
    │            ├─ name.endswith(".weight_scale_inv")
    │            │   └─ 暂存，等待配对 weight
    │            ├─ dtype 是 float8 AND name.endswith(".weight")
    │            │   └─ 暂存，等待配对 scale
    │            └─ 配对成功 → _dequantize_fp8_block_weight()
    │                            ↓
    │                fp8_weight.to(bf16) * expanded_scale.to(bf16)
    │                            ↓
    │                yield (name, bf16_weight)  ← 反量化后的 BF16 权重
    │
    └─ original_load_weights(self, 反量化后的 weights 流)
```

### 反量化公式

权重按 `(128, 128)` block 分块量化：

```python
def _dequantize_fp8_block_weight(fp8_weight, weight_scale_inv, block_size=(128,128)):
    # weight_scale_inv shape: (n_tiles, k_tiles)
    # 1. 把 scale 扩展到和 weight 一样大
    expanded_scale = weight_scale_inv.repeat_interleave(128, dim=0)  # 行方向每128展开
    expanded_scale = expanded_scale.repeat_interleave(128, dim=1)    # 列方向每128展开
    expanded_scale = expanded_scale[:n, :k].to(torch.bfloat16)
    # 2. FP8 → BF16 × scale
    return fp8_weight.to(torch.bfloat16) * expanded_scale
```

等价公式（per-block dequant）：

$$
W_{ij}^{\text{bf16}} = W_{ij}^{\text{fp8}} \cdot \text{scale\_inv}_{\lfloor i/128 \rfloor, \lfloor j/128 \rfloor}
$$

### 时序

```
┌──────────┐     ┌──────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│ Worker   │     │ _patched_load_weights │     │ _fp8_dequant         │     │ _original          │
│          │     │ (MiniMaxM2Model)      │     │ _weight_iter         │     │ _load_weights      │
└────┬─────┘     └──────────┬───────────┘     └──────────┬───────────┘     └─────────┬──────────┘
     │                      │                            │                          │
     │ load_weights(w)      │                            │                          │
     │─────────────────────▶│                            │                          │
     │                      │                            │                          │
     │                      │ quant=fp8 + npu?           │                          │
     │                      │──── YES ──────────────────▶│                          │
     │                      │                            │                          │
     │                      │                            │ 遍历 weights:             │
     │                      │                            │ "layer.weight" (fp8)     │
     │                      │                            │  → 暂存 pending           │
     │                      │                            │ "layer.weight_scale_inv"  │
     │                      │                            │  → 配对 → dequant         │
     │                      │                            │  → yield (name, bf16_w)   │
     │                      │                            │                          │
     │                      │ 反量化后的 weights 流       │                          │
     │                      │──────────────────────────────────────────────────────▶│
     │                      │                            │                          │
     │                      │          return loaded_params                          │
     │                      │◀───────────────────────────────────────────────────────│
```

---

## 5. Patch 4: MiniMaxM2Model.forward + MiniMaxM2ForCausalLM — Eagle3

### 被 Patch 的函数

**`MiniMaxM2Model.forward`**（`minimax_m2.py:373-409`）— 原始已支持 `aux_hidden_states`（通过 `_maybe_add_hidden_state`），但依赖 `SupportsEagle3` 接口方法。

### Patch 内容

**a) `MiniMaxM2Model.forward` 替换：**

```python
def _patched_minimax_m2_forward(self, input_ids, positions, ...):
    aux_layers = getattr(self, "aux_hidden_state_layers", ()) or ()
    if not aux_layers:
        return _original_forward(self, ...)
    # 手动遍历 layers，在 aux_layers 指定的层收集 hidden_states
    for idx, layer in enumerate(self.layers[start:end]):
        layer_idx = self.start_layer + idx
        if layer_idx in aux_layers:
            aux_hidden_states.append(hidden_states + residual)
        hidden_states, residual = layer(positions, hidden_states, residual)
    # 返回 (final_hidden_states, aux_hidden_states)
```

原始 `forward` 依赖 `_maybe_add_hidden_state` helper（基于 `SupportsEagle3` 基类），Patch 版本手动实现同等逻辑以兼容不同的上游版本。

**b) `MiniMaxM2ForCausalLM` 方法注入：**

```python
# 设置要收集辅助 hidden states 的层号
MiniMaxM2ForCausalLM.set_aux_hidden_state_layers = _set_aux_hidden_state_layers

# 默认的 Eagle3 aux 层: (2, mid, last-3)
MiniMaxM2ForCausalLM.get_eagle3_default_aux_hidden_state_layers = ...  # → (2, n//2, n-3)
MiniMaxM2ForCausalLM.get_eagle3_aux_hidden_state_layers = ...

# SupportsEagle3 协议所需属性
MiniMaxM2ForCausalLM.has_own_lm_head = False
MiniMaxM2ForCausalLM.has_own_embed_tokens = False
MiniMaxM2ForCausalLM.supports_eagle3 = True
```

### 时序（Eagle3 投机解码）

```
┌──────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
│ vLLM     │     │ MiniMaxM2ForCausalLM    │     │ MiniMaxM2Model       │
│ Engine   │     │ (patched)               │     │ (patched forward)    │
└────┬─────┘     └────────────┬────────────┘     └───────────┬──────────┘
     │                        │                             │
     │ set_aux_hidden_state   │                             │
     │   _layers((2,12,21))   │                             │
     │───────────────────────▶│                             │
     │                        │ model.aux_hidden_state      │
     │                        │   _layers = (2,12,21)       │
     │                        │────────────────────────────▶│
     │                        │                             │
     │ forward(...)           │                             │
     │───────────────────────▶│                             │
     │                        │ model.forward(...)          │
     │                        │────────────────────────────▶│
     │                        │                             │
     │                        │                             │ aux_layers = (2,12,21)
     │                        │                             │ for each layer:
     │                        │                             │   if idx in aux_layers:
     │                        │                             │     collect hidden_states
     │                        │                             │   forward layer
     │                        │                             │
     │                        │   return (final, [aux2,      │
     │                        │            aux12, aux21])    │
     │                        │◀────────────────────────────│
     │                        │                             │
     │ return hidden_states,  │                             │
     │        aux_hidden_states                             │
     │◀───────────────────────│                             │
     │                        │                             │
     │ Eagle3 draft model     │                             │
     │ uses aux_hidden_states │                             │
     │ as cross-attention     │                             │
     │ context                │                             │
```

---

## 6. 四 Patch 关系总结

```
MiniMax-M2 模型加载 & 推理流程:
══════════════════════════════════════════════════════════════════

加载阶段:
  checkpoint (fp8) ──Patch 3──▶ BF16 weights ──▶ original load_weights
                                (fp8 dequant)

模型构造:
  MiniMaxM2Attention.__init__ ──Patch 2a──▶ num_kv_head_replicas
                                            + k_norm sharding

推理阶段:
  hidden_states
      │
      ├─ MiniMaxM2Attention.forward ──Patch 2b──▶ 融合 QKV+Norm+RoPE
      │       │
      │       ├─ MiniMaxM2MoE.forward ──Patch 1──▶ TP all_reduce
      │       │
      │       └─ MiniMaxText01RMSNormTP ──Patch (linear_attn)──▶ NPU fast path
      │
      └─ MiniMaxM2Model.forward ──Patch 4──▶ aux_hidden_states (Eagle3)
```
