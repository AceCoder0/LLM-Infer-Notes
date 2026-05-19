# patch_gqa_c8.py 分析

> 文件：`vllm-ascend/vllm_ascend/patch/worker/patch_gqa_c8.py`
> 分析日期：2026-05-19

## 1. 总览

该 patch 在 worker 启动时通过 import 自动执行，无需显式调用。它 monkey-patch 了 Qwen3 和 GLM4-MoE 两个模型的 `load_weights` 方法，目的是拦截 C8（W8A8C8，即权重 INT8 + 激活 INT8 + KV Cache INT8）量化模型 checkpoint 中额外的 KV cache scale/offset 参数，让它们正确加载到 vLLM 的参数中，而不是被 `AutoWeightsLoader` 丢弃。

## 2. Patch 加载链

```
vllm_ascend/patch/worker/__init__.py (line 55)
    └── import vllm_ascend.patch.worker.patch_gqa_c8  (模块级代码立即执行)
            ├── 保存原始方法引用
            └── 替换为 lambda 包装的 patched 函数
```

## 3. 被 Patch 的目标函数

| 原始函数 | 文件位置 | 调用链 |
|---------|---------|--------|
| `Qwen3ForCausalLM.load_weights` | `vllm/model_executor/models/qwen3.py:335` | `AutoWeightsLoader.load_weights(weights)` |
| `Glm4MoeForCausalLM.load_weights` | `vllm/model_executor/models/glm4_moe.py:703` | `AutoWeightsLoader.load_weights(weights)` |

两个原始方法都是简单的委托，核心逻辑在 `AutoWeightsLoader` 中（vllm 内置）。`AutoWeightsLoader` 通过参数名匹配来自动加载权重，对于不认识的参数名会**静默丢弃**。

C8 量化模型 checkpoint 中包含额外的 KV cache scale/offset 张量（如 `k_cache_scale`、`v_cache_offset`），这些张量的命名规范与 vLLM 内置参数不匹配，需要通过 patch 拦截。

## 4. 时序图

```
┌──────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Worker   │     │ patch_gqa_c8.py  │     │ Qwen3ForCausalLM  │     │ _patched_causal_   │     │ AutoWeightsLoader │
│ __init__ │     │  (模块加载)       │     │ (被 patch 后)      │     │ lm_load_weights    │     │ (原始权值加载器)   │
└────┬─────┘     └────────┬─────────┘     └────────┬──────────┘     └─────────┬──────────┘     └────────┬─────────┘
     │                    │                        │                          │                        │
     │ import worker/     │                        │                          │                        │
     │   __init__.py      │                        │                          │                        │
     │───────────────────▶│                        │                          │                        │
     │                    │                        │                          │                        │
     │                    │ 1. 保存原始方法:         │                          │                        │
     │                    │ _orig = Qwen3...        │                          │                        │
     │                    │   .load_weights         │                          │                        │
     │                    │                        │                          │                        │
     │                    │ 2. 替换:                │                          │                        │
     │                    │ load_weights =          │                          │                        │
     │                    │   lambda self, w:       │                          │                        │
     │                    │   _patched(...,_orig)   │                          │                        │
     │                    │───────────────────────▶│                          │                        │
     │                    │                        │                          │                        │
     │                    │                        │ (后续权重加载)              │                        │
     │                    │                        │                          │                        │
     ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
     │  模型加载阶段                                                                                           │
     ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
     │                    │                        │                          │                        │
     │                    │                        │ load_weights(weights)    │                        │
     │                    │                        │─────────────────────────▶│                        │
     │                    │                        │                          │                        │
     │                    │                        │                          │ 3. 检查 quant_config    │
     │                    │                        │                          │    有 get_cache_scale?  │
     │                    │                        │                          │                        │
     │                    │                        │                          │ 4. 创建拦截通道         │
     │                    │                        │                          │ _intercept_c8_scales()  │
     │                    │                        │                          │                        │
     │                    │                        │                          │ 5. 遍历 weights:        │
     │                    │                        │                          │───────────────────────▶│
     │                    │                        │                          │                        │
     │                    │                        │                          │    for name, w in       │
     │                    │                        │                          │    weights:             │
     │                    │                        │                          │      scale_name =       │
     │                    │                        │                          │      quant_config       │
     │                    │                        │                          │      .get_cache_scale   │
     │                    │                        │                          │      (name)             │
     │                    │                        │                          │                        │
     │                    │                        │                          │    ┌── 6a. 是 scale: ──┐│
     │                    │                        │                          │    │ 直接加载到对应      ││
     │                    │                        │                          │    │ nn.Parameter        ││
     │                    │                        │                          │    │ ↓                   ││
     │                    │                        │                          │    │ c8_loaded_params     ││
     │                    │                        │                          │    │ .add(scale_name)    ││
     │                    │                        │                          │    │ ↓                   ││
     │                    │                        │                          │    │ 不 yield (丢弃)     ││
     │                    │                        │                          │    └───────────────────┘│
     │                    │                        │                          │                        │
     │                    │                        │                          │    ┌── 6b. 普通权重: ─┐│
     │                    │                        │                          │    │ yield name, w       ││
     │                    │                        │                          │    └───────────────────┘│
     │                    │                        │                          │                        │
     │                    │                        │                          │ 7. original_load_weights │
     │                    │                        │                          │    (_intercept_gen)       │
     │                    │                        │                          │────────────────────────▶│
     │                    │                        │                          │                        │
     │                    │                        │                          │    ← 返回 loaded_params │
     │                    │                        │                          │                        │
     │                    │                        │                          │ 8. loaded_params         │
     │                    │                        │                          │    .update(              │
     │                    │                        │                          │     c8_loaded_params)    │
     │                    │                        │                          │                        │
     │                    │                        │     return loaded_params  │                        │
     │                    │                        │◀─────────────────────────│                        │
     │                    │                        │                          │                        │
```

## 5. 代码执行流程

### 5.1 模块加载时（import 阶段）

```python
# vllm_ascend/patch/worker/patch_gqa_c8.py

# Step 1: 保存原始方法引用
_orig_qwen3_causal_lm_load_weights = Qwen3ForCausalLM.load_weights
_orig_Glm4_causal_lm_load_weights = Glm4MoeForCausalLM.load_weights

# Step 2: 用 lambda 替换，lambda 内调用 patched 函数 + 原始方法
Qwen3ForCausalLM.load_weights = lambda self, weights: _patched_causal_lm_load_weights(
    self, weights, _orig_qwen3_causal_lm_load_weights
)
Glm4MoeForCausalLM.load_weights = lambda self, weights: _patched_causal_lm_load_weights(
    self, weights, _orig_Glm4_causal_lm_load_weights
)
```

### 5.2 模型加载时（load_weights 被调用）

```
_patched_causal_lm_load_weights(self, weights, original_load_weights)
    │
    ├─ 判断: quant_config 存在 且 有 get_cache_scale 方法?
    │   └─ NO  → 直接调用 original_load_weights(self, weights)
    │   └─ YES → 进入拦截流程 ↓
    │
    ├─ 获取 params_dict = dict(self.named_parameters())
    ├─ 初始化 c8_loaded_params = set()
    │
    ├─ 创建 _intercept_c8_scales 生成器
    │   └─ for name, loaded_weight in weights:
    │       ├─ scale_name = quant_config.get_cache_scale(name)
    │       ├─ if scale_name is not None AND scale_name in params_dict:
    │       │   ├─ param = params_dict[scale_name]
    │       │   ├─ weight_loader(param, loaded_weight.squeeze())
    │       │   └─ c8_loaded_params.add(scale_name)
    │       │       # 注意: 不 yield → 这个权重被"吃掉"了
    │       └─ else:
    │           └─ yield name, loaded_weight  # 正常传递
    │
    ├─ loaded_params = original_load_weights(self, _intercept_c8_scales(weights))
    │   # AutoWeightsLoader 只看到 "经过滤" 的权重流，
    │   # C8 scale/offset 张量已被拦截移除
    │
    └─ loaded_params.update(c8_loaded_params)
       return loaded_params
```

## 6. get_cache_scale 名称映射

`get_cache_scale` 定义在量化配置类上（`Fp8Config` / `QuarkConfig`），负责 checkpoint 参数名 → vLLM 内部参数名的映射：

| Checkpoint 名称 (输入) | vLLM 内部参数名 (输出) | 意义 |
|------------------------|----------------------|------|
| `xxx.k_proj.output_scale` | `xxx.attn.k_scale` | K 通道量化 scale |
| `xxx.v_proj.output_scale` | `xxx.attn.v_scale` | V 通道量化 scale |
| `xxx.q_proj.output_scale` | `xxx.attn.q_scale` | Q 通道量化 scale |
| `xxx.self_attn.prob_output_scale` | `xxx.attn.prob_scale` | Attention prob 量化 scale |
| 其他名称 | `None` | 不是 C8 参数，正常传递 |

## 7. 为什么需要这个 Patch

C8 (W8A8C8) 量化模式下，KV cache 使用 INT8 存储。推理时需要 per-channel 的 scale 和 offset 来反量化：

```
kv_fp = kv_int8 * scale + offset
```

这些 scale/offset 是模型 checkpoint 的一部分，但：
- Checkpoint 中的命名（如 `k_cache_scale`）不匹配 vLLM `AutoWeightsLoader` 的参数匹配规则
- `AutoWeightsLoader` 对不认识的参数会静默丢弃（不报错，但模型精度会崩）

Patch 的解决方案：在权重流到达 `AutoWeightsLoader` 之前截获这些特殊参数，手动路由到正确的 `nn.Parameter`。

## 8. 相关 Patch

- **`patch_weight_utils.py`**: 类似目的，处理 DeepSeek V2 C8 模型的 `maybe_remap_kv_scale_name`，增强了 KV scale 名称的重映射逻辑
- 两个 patch 都是为解决同一类问题：量化 checkpoint 中的 KV cache 参数命名与 vLLM 内部不一致
