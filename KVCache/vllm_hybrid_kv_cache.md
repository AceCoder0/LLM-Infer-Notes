# vLLM 混合 KV Cache 管理（DeepSeek V4 视角）

> 核心问题：一个模型里不同层需要**不同类型 / 不同 page size** 的 KV Cache（滑动窗口、压缩 kv、indexer kv、compressor 暂存态……），vLLM 怎么规划、分配、存储它们？
>
> 一句话答案：**所有 cache 组被"打包"进同一个 backing buffer（slab），每层按 `offset` / `block_stride` 切成错位视图；凡是"各自组里从 offset 0 累加 page_size、落点相同"的层共享同一个 KVCacheTensor（同一块 slab）。**

---

## 1. 问题背景：DSV4 的异构 KV Cache

### 1.1 整体架构（mHC）

DeepSeek V4 在 attention + MoE 之前先用 `hc_pre` 把 hidden states 降维，计算后再用 `hc_post` 升维、与残差线上的 hidden states concat。主体 attention / MoE 计算量不增加，同时保留更多前层信息。

### 1.2 Attention 层的压缩

- 传统 MLA 对所有 token 存 KV；DSV4 引入 **kv 压缩**（compressor 模块，可训练参数），两种压缩比：**4** 和 **128**，交替出现在各层。
- **c4 层 = CSA（Compressed Sparse Attention）**：kv 分三股——
  - **sliding window kv**：固定窗口内的 kv；
  - **compressed kv**：经 compressor 压缩后的 kv（不自接参与 attention，等 top-k 选中）；
  - **indexer kv**：用 compressor 压缩后得到低秩 key，与低秩 query 做 MQA 得 indexer scores，选出 compressed kv 的 top-k indices（同 DSA）。
  - 最终 = sliding window kv ⊕ selected compressed kv，喂给 MQA。
- **c128 层 = HCA（Heavily Compressed Attention）**：去掉 indexer 稀疏化，只保留 sliding window + 主 compressor。
- **不压缩的层**（Flash 的 L0/L1、MTP）：只剩 sliding window kv。

### 1.3 每层出现的 cache 一览

| 层 | cache 类型 |
| --- | --- |
| 不压缩层 | swa cache |
| c4 层 | swa cache、compressed kv cache、compressed indexer kv cache、c4a kv state、c4a score state、c4i kv state、c4i score state |
| c128 层 | swa cache、compressed kv cache、c128a kv state、c128a score state |

> c4a = c4 的 attention，c4i = c4 的 indexer。compressor 不满压缩比时需要暂存 kv 和 softmax 加权结果（score），所以每个 compressor 还带两个暂存 cache（kv state / score state）。

**DeepSeek V4 Flash 的 43 层结构**：

```
层 0 / 1        : swa（不压缩）
层 2 ~ 42 偶数层 : c4（CSA，压缩比 4）
层 2 ~ 42 奇数层 : c128（HCA，压缩比 128）
```

---

## 2. 参考文档

- 官方 vLLM 博客（DeepSeek V4 in vLLM）：https://vllm.ai/blog/2026-04-24-deepseek-v4
- vLLM Hybrid KV Cache Manager 设计文档：https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/
- mengqingcao 博客（DeepSeek V4 KVCache 管理）：https://mengqingcao.github.io/posts/deepseek-v4-kvcache-management/
- DeepSeek V4 论文：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf

对应源码：

- `vllm/v1/core/kv_cache_utils.py` —— 分组 + 打包布局的核心
- `vllm/v1/kv_cache_interface.py` —— Spec / Tensor 数据类
- `vllm/v1/core/kv_cache_coordinator.py` —— 共享 BlockPool
- `vllm/v1/worker/gpu_model_runner.py` —— 运行时 tensor 初始化
- `vllm/v1/worker/gpu/attn_utils.py` —— 视图 reshape
- `vllm/models/deepseek_v4/{attention,compressor}.py` —— DSV4 模型侧 spec 产出

---

## 3. vLLM 的三个设计决策（官方博客）

1. **单一逻辑 block size 256**：管理面看到的 block_size 统一为 256。
2. **compressor 状态当作 sliding-window KV**：compressor 的暂存态复用 `SlidingWindowMLASpec`，天然支持窗口式淘汰。
3. **page size 统一成几个桶 + 共享 block pool**：异构 page size 归类成组，所有组共享一个 BlockPool。

---

## 4. 关键概念（数据类）

| 概念 | 位置 | 说明 |
| --- | --- | --- |
| `KVCacheGroupSpec` | kv_cache_interface.py:950 | 一组共享**同一个 block table** 的层；在 manager 眼里这组算"一层"。字段：`layer_names`、`kv_cache_spec`、`is_eagle_group` |
| `UniformTypeKVCacheSpecs` | kv_cache_interface.py:829 | 多个同类型层的集合。`page_size_bytes` = 各层之和（:840）；`get_num_layer_tuples` = 众数 page size 的层数（:881，DSV4 专用） |
| `MLAAttentionSpec` | kv_cache_interface.py:381 | 完整 MLA / DSV4 主 attention。`storage_block_size = block_size // compress_ratio`（:396）；`real_page_size_bytes` 对 fp8_ds_mla = `storage_block_size × 584`（:400） |
| `SlidingWindowMLASpec` | kv_cache_interface.py:623 | 滑动窗口 + compressor 暂存态。merge 要求同 `sliding_window`（:665） |
| `KVCacheTensor` | kv_cache_interface.py:938 | `size`、`shared_by: list[str]`、`offset`、`block_stride`（>0 表示 packed） |

---

## 5. 分组：按什么准则分 group

入口 `group_and_unify_kv_cache_specs`（kv_cache_utils.py:1592）：

```python
# DSV4 专属分组：
# - 所有 MLAAttentionSpec          → 归进 1 个 group（full_mla，主 attention 全家桶）
# - 所有 SlidingWindowMLASpec      → 按 (block_size, sliding_window) 分组
return [mla_uniform_spec, *swa_uniform_specs]
```

**分组准则**：同一组内所有层必须"需要同样多的 token slots"（同类型）。所以主 attention 不管压缩比多少都进一组（因为它们按 token 一个 slot 一个 slot 存），滑动窗口/compressor 态则要窗口大小一致才能共用 block table。

### 5.1 层 tuple 对齐

`_get_kv_cache_groups_uniform_groups`（:1670）用 `_approximate_gcd`（:1635，暴力枚举 d 最小化向上 padding）选一个"层 tuple 数"，把每组层数对齐：

```python
num_layer_tuples = _approximate_gcd(
    num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
)
num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in ...]
```

> 层 tuple = "跨组的一竖列"。例：11 个 C4 层 + 10 个 C128 层，可以定义层 tuple = [C4I, C4A, C128]，full_mla 组就含 11 个 tuple（代码注释原话）。

对齐的目的是：**每组层数一致 → slab 的"行结构"固定、可重复**。每个 block id 的 slab 形状都一样，只是装的数据随 block 变。

### 5.2 DSV4 Flash 的 5 个组

| group | block_size | 内容 |
| --- | --- | --- |
| G1 | 256 | full_mla（c4a + c4i + c128a 主 kv） |
| G2 / G3 | 256 | swa / compressor 态 |
| G4 | 8 | 压缩后小 page 的 sliding window（ratio 128） |
| G5 | 4 | 压缩后小 page 的 sliding window（ratio 4） |

（对应博客"5 组、block size 256/256/256/8/4、22 个层 tuple"。）

---

## 6. 核心：谁和谁组成一个 slab

### 6.1 判定逻辑

`_get_packed_kv_cache_layout`（kv_cache_utils.py:1283）是唯一裁决者：

```python
def _get_packed_kv_cache_layout(kv_cache_groups):
    """Lay out each cache group densely in one shared block slab.

    A block ID is owned by one cache group at a time, so layouts from different
    groups may overlap. Layers within a group remain disjoint.
    """
    layers_by_offset: dict[int, list[str]] = defaultdict(list)
    block_stride = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        byte_offset = 0
        for layer_name in group.layer_names:
            if isinstance(spec, UniformTypeKVCacheSpecs):
                page_size = spec.kv_cache_specs[layer_name].page_size_bytes
            else:
                page_size = spec.page_size_bytes
            layers_by_offset[byte_offset].append(layer_name)
            byte_offset += page_size
        block_stride = max(block_stride, byte_offset)
    assert block_stride > 0
    return block_stride, layers_by_offset
```

**要点**：

- 每个组**独立**从 `byte_offset = 0` 开始，把该组的层一层一层往下放，每层累加自己的 `page_size`；
- `layers_by_offset[byte_offset].append(layer_name)` —— 落点 offset 相同的层进同一个列表 → **组成同一个 KVCacheTensor（同一块 slab）**；
- `block_stride` = 所有组里累计最长的那个 → 每个 block（slab）的宽度；
- **组间 offset 不需要对齐**——撞上就共享，撞不上就各自独占一个 tensor。安全性靠 docstring 的不变量：*一个 block id 同一时刻只被一个 cache group 占用*。

### 6.2 具体例子（3 个组）

```
G1: 每层 100 字节，共 2 层
G2: 每层 100 字节，共 2 层
G3: 每层  50 字节，共 4 层
```

摆放（各自从 0 累加）：

```
G1:  a──@0──  b──@100──       合计 200
G2:  a──@0──  b──@100──       合计 200
G3:  a@0  b@50  c@100  d@150  合计 200
```

按 offset 归并：

```
offset   0 : G1.a, G2.a, G3.a   → tensor#1（3 层共享）
offset  50 : G3.b               → tensor#2（只有 G3）
offset 100 : G1.b, G2.b, G3.c   → tensor#3（3 层共享）
offset 150 : G3.d               → tensor#4（只有 G3）
block_stride = 200
```

内存图：

```
┌──────────────── slab @ block B（宽 200 字节）────────────────┐
│  0 .. 100 :  G1.a  G2.a  G3.a      ← tensor#1               │
│ 50 .. 100 :  G3.b                  ← tensor#2               │
│ 100 .. 200 :  G1.b  G2.b  G3.c     ← tensor#3               │
│ 150 .. 200 :  G3.d                 ← tensor#4               │
└─────────────────────────────────────────────────────────────┘
```

这 4 个 tensor 实际是**同一块 backing buffer 的 4 个错位视图**（alias）。`_get_kv_cache_config_packed`（:1330）为每个 distinct offset 生成一个 `KVCacheTensor(size=total_size, shared_by=..., offset=..., block_stride=...)`。

### 6.3 DSV4 Flash 的真实数字

- 每个组都从 0 开始 → **offset 0 上永远有每个组的第一层**，所以第一个 tensor 的 `shared_by` 有 5 项（每组一层）；
- 博客数据：22 个层 tuple × 3 个 offset = **66 个 KVCacheTensor**（具体数字取决于真实每层 page size，机制与上面 3 组例子完全一样）；
- 258 token 请求的分配：`cdiv(258,256)=2`（G1/G2/G3）、`cdiv(258,8)=33`（G4）、`cdiv(258,4)=65`（G5）。G4/G5 前面 32/64 个 block 被 `blockpool.free_blocks` 立即释放（滑动窗口），实际占用 1 个。

---

## 7. 运行时怎么用

### 7.1 一个 backing buffer，多个错位视图

`gpu_model_runner.py` 的 `initialize_kv_cache_tensors`：若 `kv_cache_tensor.block_stride > 0`，只分配一次 `packed_backing`，其余 tensor 全部 alias 它。

`attn_utils.py` 的 `_reshape_attention_kv_cache` 切视图：

```python
kv_raw_tensor.view(-1, block_stride)[:, offset:offset + page_bytes].view(dtype).view(shape)
```

### 7.2 动态分配（序列变长时）

- `num_blocks` = `available_memory // block_stride`，是**共享的固定容量**（由 max_concurrency 控制）；
- block id 是共享 BlockPool 里的**句柄**；每个组有自己的 block table，各自按需分配；
- 序列变长 → 该组的 block table 追加新的 block id；**一个 block id 同一时刻只被一个组用**，所以不同组的 block 在 slab 里错位重叠是安全的。

### 7.3 分片大小怎么保证均匀

- piece = `block_size × kv_hidden_size`（每层）；管理面看到的 block_size 被强制统一（设计决策 1）；
- 层 tuple 对齐保证每组层数一致 → slab 结构固定；
- DSV4 的 page size 无法对齐 → 走 packed 布局（每层独立 offset），绕开对齐浪费。

---

## 8. 简单模型对比（Llama / Qwen3）

- 1 个 group、N 层 → **N 个独立 tensor**，每个 `page_size × num_blocks`；
- `get_num_blocks`（:993）= `available_memory // page_size // num_layers`；
- 所有层共享**一个** block table。没有打包、没有错位视图。

| | 简单模型（Llama/Qwen3） | DSV4 |
| --- | --- | --- |
| 组数 | 1 | 5（Flash） |
| tensor 数 | N（每层一个） | 66（22 tuple × 3 offset） |
| 每层位置 | 独占 tensor | 共享 slab 的错位视图 |
| block table | 全部层共用 1 个 | 每组 1 个 |

---

## 9. GPU vs Ascend 差异

| | vLLM（GPU） | vLLM Ascend |
| --- | --- | --- |
| 压缩维度 | **block_size 维**：block 被压成 `block_size // compress_ratio` 个 slots | **num_blocks 维**：分配前先把 `num_tokens` 压成 `num_tokens // compress_ratio` |
| 压缩后 block_size | 管理面 256，kernel 侧变小 | 管理面和 kernel 侧都是原始 block_size |
| slot_mapping | `get_compressed_slot_mapping`（compressor_utils.py），-1 位置跳过 | 压缩 position id + 压缩后的 slot mapping |
| 适配点 | — | `vllm_ascend.utils.get_compressed_pos_and_indices`、`MultiGroupBlockTable.compute_slot_mapping` |

> Ascend 限制：attention 算子最小 block_size 16，性能推荐 128。所以 c128a 在 GPU 上一个小 tensor 就够，Ascend 上太大 → 让它也成为一个与其他组共享 tensor 的组（消除大量 pad block），代价是 KV Cache Planning 与 vLLM 略有不同。
>
> 注：block_size 的实验仍在进行（权衡 prefix cache hit rate 与算子性能），未来方案可能变化。
