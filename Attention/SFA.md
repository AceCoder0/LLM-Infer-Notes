# MLA

> 用于模型：DeepSeekV2 DeepSeekV3 DeepSeekV3.1

## 计算原理

### 总体框架

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/27897e7a-c04f-462f-b85e-a12f1217b05a?document_id=91241147-c04b-4ee2-adce-7d5a741a4970 "")

### 计算公式解析

### DeepSeek-V3 MLA 计算公式整理

#### 1. 符号说明

| 符号  | 含义                       | 符号            | 含义                                     |
| ----- | -------------------------- | --------------- | ---------------------------------------- |
| $h_t$ | 第 $t$ 个 token 的输入向量 | $n_h$           | 注意力头数                               |
| $d$   | 嵌入维度                   | $d_c, d'_c$     | KV 与 Query 压缩维度                     |
| $d_h$ | 单头维度                   | $d_h^R$         | RoPE 维度                                |
| $W$   | 投影矩阵                   | $[\cdot;\cdot]$ | 向量拼接                                 |
| $c$   | 压缩后的潜向量             | $C, R$          | 内容 (Content) 与 旋转 (Rotary) 部分 |

***

#### 2. KV 生成与压缩 (Key-Value Generation)

仅缓存 $c_t^{KV}$ 和 $k_t^R$ 以节省显存。

$$
\begin{aligned}
c_t^{KV} &= W_{DKV} h_t & \quad & \text{(KV 压缩)} \\
{[}k_{t,1}^C; \dots; k_{t,n_h}^C{]} = k_t^C &= W_{UK} c_t^{KV} & \quad & \text{(内容 Key 解压缩)} \\
k_t^R &= \text{RoPE}(W_{KR} h_t) & \quad & \text{(旋转 Key，多头共享)} \\
k_{t,i} &= {[}k_{t,i}^C; k_t^R{]} & \quad & \text{(完整 Key)} \\
{[}v_{t,1}^C; \dots; v_{t,n_h}^C{]} = v_t^C &= W_{UV} c_t^{KV} & \quad & \text{(Value 解压缩)}
\end{aligned}
$$

***

#### 3. Query 生成与压缩 (Query Generation)

$$
\begin{aligned}
c_t^Q &= W_{DQ} h_t & \quad & \text{(Query 压缩)} \\
{[}q_{t,1}^C; \dots; q_{t,n_h}^C{]} = q_t^C &= W_{UQ} c_t^Q & \quad & \text{(内容 Query 解压缩)} \\
{[}q_{t,1}^R; \dots; q_{t,n_h}^R{]} = q_t^R &= \text{RoPE}(W_{QR} c_t^Q) & \quad & \text{(旋转 Query，分头计算)} \\
q_{t,i} &= {[}q_{t,i}^C; q_{t,i}^R{]} & \quad & \text{(完整 Query)}
\end{aligned}
$$

***

#### 4. 注意力计算 (Attention Calculation)

$$
\begin{aligned}
o_{t,i} &= \sum_{j=1}^{t} \text{Softmax}_j \big( \frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}} \big) v_{j,i}^C & \quad & \text{(单头注意力输出)} \\
u_t &= W_O {[}o_{t,1}; \dots; o_{t,n_h}{]} & \quad & \text{(最终输出投影)}
\end{aligned}
$$

#### 计算图

> 什么是吸收（MQA）？什么是非吸收（MHA）？
> ![](../assets/mha-vs-mqa.png)

##### 非吸收的MLA （MHA版本）

![](../assets/sfa-mha.png)

##### 吸收的MLA （MQA版本）

![](../assets/sfa-mqa.png)

## MindIE-LLM实现

> 待补充

# SFA

> 用于DeepSeekV3.2，GLM5等

![](../assets/sfa-article.png)

## 计算原理

仅仅比MLA多一个Lightning Indexer

### Lightning Indexer 计算公式整理

#### 1. 符号说明

| 符号           | 含义                                     | 符号      | 含义             |
| -------------- | ---------------------------------------- | --------- | ---------------- |
| $h_t, h_s$     | 当前 Query  token 与 preceding Key token | $H_I$     | Indexer 头数     |
| $d_I$          | Indexer 投影维度                         | $I_{t,s}$ | Token 间索引分数 |
| $q^I, k^I$     | Indexer 查询与键向量                     | $w^I$     | Indexer 标量权重 |
| $c_s$          | 压缩后的 KV 条目                         | $u_t$     | 注意力输出       |
| $\text{Top-k}$ | 前 $k$ 个高分选择                        | $\cdot$   | 点积或标量乘法   |

***

#### 2. 索引分数计算 (Index Score Calculation)

计算当前 token $t$ 与历史 token $s$ 之间的相关性分数。

$$
\begin{aligned}
I_{t,s} &= \sum_{j=1}^{H_I} w_{t,j}^I \cdot \text{ReLU}( q_{t,j}^I \cdot k_s^I )
\end{aligned}
$$

* $q_{t,j}^I, w_{t,j}^I$ 源自 $h_t$；$k_s^I$ 源自 $h_s$。

* 采用 ReLU 激活函数以提升吞吐量。

***

#### 3. 稀疏选择与注意力 (Sparse Selection & Attention)

根据索引分数筛选 Top-k 个 KV 条目进行注意力计算。

$$
\begin{aligned}
\mathcal{S}_t &= \{ s \mid I_{t,s} \in \text{Top-k}(I_{t,:}) \} & \quad & \text{(选中 token 集合)} \\
u_t &= \text{Attn}( h_t, \{ c_s \mid s \in \mathcal{S}_t \} ) & \quad & \text{(稀疏注意力输出)}
\end{aligned}
$$

* 仅对索引分数最高的 $k$ 个历史 token 对应的 $c_s$ 进行注意力运算。

* 支持 FP8 实现以保证计算效率。

> 当前vllm和MindIE的代码实现只有吸收（MQA）版本，但是ds官方代码仓有非吸收（MHA）版本和吸收（MQA）版本

![](../assets/sfa-mqa.png)

## MindIE-LLM实现

> dsv32实现只有吸收版本

### 代码实现的计算图

![](../assets/sfa-mqa.png)

### 关键代码

> 仅展示关键代码 具体代码在mindie\_llm\\runtime\\layers\\attention\\backend\\sparse\_attention.py

##### q k v的preprocess

```python
def sfa_preprocess(...):
    ...
    ckq = self.q_a_proj(hidden_states) # 对应q_c计算
    q_c = self.q_a_layernorm(ckq)
    ...
    kv_no_split = self.kv_a_proj_with_mqa(hidden_states)
    ...

def sfa_prefill_preprocess(...):
    # =============q相关的计算================
    # 这里decode 命名不好，明明是prefill
    decode_q = self.q_b_proj(q_c) # q_b_proj
    bsz, _ = decode_q.shape
    decode_q = decode_q.view(bsz, self.num_heads_per_rank, 1, self.qk_head_dim) # 分head
    decode_q_nope, decode_q_pe = torch.split(
        decode_q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    ) # split nope和rope
    decode_q_nope = decode_q_nope.view(-1, self.num_heads_per_rank, self.qk_nope_head_dim).transpose(0, 1) # 转置为(head_num, bsz, nope_dim) 这里转置是为了后面分别对每个head做matmul
    decode_q_nope = (
        torch.matmul(decode_q_nope, self.kv_b_proj_w_k) # 这里的kv_b_proj_w_k明明不好，明明是对q做投影
        .transpose(1, 0)
        .view(bsz, 1, self.num_heads_per_rank, self.kv_lora_rank)
    ) # 转置回来
    ...
    # 下面几行代码应该放在上面，q的计算放在一起
    decode_q_pe = torch_npu.npu_interleave_rope(decode_q_pe,
                                                attn_metadata.cos_table,
                                                attn_metadata.sin_table)

    decode_q_nope = decode_q_nope.view(bsz, self.num_heads_per_rank, self.kv_lora_rank)
    decode_q_pe = decode_q_pe.view(bsz, self.num_heads_per_rank, -1)

    # =================kv相关的计算=================
    decode_k_rope, decode_k_nope, k_rope_a, k_nope_a = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos_cache,
            sin_cache,
            attn_metadata.slot_mapping.to(torch.int64),
            self.pe_cache,
            self.kv_cache,
            c_kv_scale=None,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode='PA',
            is_output_kv=is_output_kv)
    # 调用indexer_select 得到tok_indices
    topk_indices = self.indexer_select(hidden_states, q_c, forward_context, attn_metadata)
    key_states = None
    if forward_context.is_prefill and self.cp_size > 1:
        key_states = (k_nope_a, k_rope_a)
    decode_preprocess_res = PrefillSFAPreprocessResult(
        q_nope=decode_q_nope,
        q_pe=decode_q_pe,
        k_nope=decode_k_nope,
        k_pe=decode_k_rope,
        value=decode_k_nope,
        topk_indices=topk_indices,
        key_states=key_states
    )
    return decode_preprocess_res

```

##### indexer\_select计算

```python
def indexer_select(
        self,
        hidden_state: torch.Tensor,
        q_c: torch.Tensor,
        forward_context: ForwardContext, 
        attn_metadata: SfaMetadata
    ):
    q = self.indexer.wq_b(q_c) 
    q = q.view(-1, self.indexer.n_heads, self.indexer.head_dim)
    q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.indexer.head_dim - self.qk_rope_head_dim], dim=-1)

    q_pe = q_pe.unsqueeze(2)
    q_pe = torch_npu.npu_interleave_rope(q_pe,
                                            attn_metadata.cos_table,
                                            attn_metadata.sin_table)
    q_pe = q_pe.squeeze(2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    k_proj = self.indexer.wk(hidden_state)
    k = self.indexer.k_norm(k_proj).unsqueeze(1)
    k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.indexer.head_dim - self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.unsqueeze(2)
    k_pe = torch_npu.npu_interleave_rope(k_pe,
                                            attn_metadata.cos_table.view(-1, 1, 1, self.qk_rope_head_dim),
                                            attn_metadata.sin_table.view(-1, 1, 1, self.qk_rope_head_dim))
    k_pe = k_pe.squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    torch_npu.npu_scatter_nd_update_(self.index_cache.view(-1, k.shape[-1]),
                                        attn_metadata.slot_mapping.view(-1, 1), 
                                        k.view(-1, k.shape[-1]))

    weights = self.indexer.weights_proj(hidden_state)
    actual_seq_lengths_key = attn_metadata.actual_seq_lengths_kv \
        if forward_context.is_prefill else attn_metadata.seq_lens
    
    if forward_context.is_prefill and self.cp_size > 1:
        ...
    else:
        topk_indices, _ = torch_npu.npu_lightning_indexer(
            query=q,
            key=self.index_cache,
            weights=weights,
            actual_seq_lengths_query=attn_metadata.actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=attn_metadata.block_tables,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3
        )

    return topk_indices
```

##### 调用mlapo融合算子的decode preprocess

> ```python
> # 分支条件
> if not forward_context.is_prefill and self.enable_mlapo
> ```

```python
def sfa_decode_mlapo_preprocess(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata
    ):
        bsz, _ = hidden_states.shape

        decode_q_nope, cache1, decode_q_pe, cache2 = torch.ops.mie_ops.npu_mla_process(
            input=hidden_states,
            gamma0=self.mlapo_weight_pack.gamma0,
            beta0=self.mlapo_weight_pack.beta0,
            wdqkv=self.mlapo_weight_pack.wd_qkv,
            descale0=self.mlapo_weight_pack.deq_scale_qkv,
            gamma1=self.mlapo_weight_pack.gamma1,
            beta1=self.mlapo_weight_pack.beta1,
            wuq=self.mlapo_weight_pack.wu_q,
            descale1=self.mlapo_weight_pack.qb_deq_scl,
            gamma2=self.mlapo_weight_pack.gamma2,
            cos=attn_metadata.cos_table,
            sin=attn_metadata.sin_table,
            wuk=self.kv_b_proj_w_k,
            kv_cache=self.kv_cache,
            kv_cache_rope=self.pe_cache,
            slotmapping=attn_metadata.slot_mapping.flatten().to(torch.int32),
            quant_scale0=self.mlapo_weight_pack.quant_scale0,
            quant_offset0=self.mlapo_weight_pack.quant_offset0,
            bias0=self.mlapo_weight_pack.quant_bias_qkv,
            quant_scale1=self.mlapo_weight_pack.quant_scale1,
            quant_offset1=self.mlapo_weight_pack.quant_offset1,
            bias1=self.mlapo_weight_pack.qb_qt_bias,
            ctkv_scale=self.mlapo_weight_pack.ctkv_scale,
            q_nope_scale=self.mlapo_weight_pack.q_nope_scale,
            cache_mode_opt="krope_ctkv",
            quant_mode_opt="per_tensor_quant_asymm",
        )
        decode_k_nope = self.kv_cache
        decode_k_pe = self.pe_cache
        decode_q_nope = decode_q_nope.view(bsz, self.num_heads_per_rank, self.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, self.num_heads_per_rank, -1)

        topk_indices = self.indexer_select(hidden_states, q_c, forward_context, attn_metadata)
        decode_preprocess_res = DecodeSFAPreprocessResult(
            q_nope=decode_q_nope,
            q_pe=decode_q_pe,
            k_nope=decode_k_nope,
            k_pe=decode_k_pe,
            topk_indices=topk_indices
        )
        
        return decode_preprocess_res
```

##### 调用npu\_sparse\_flash\_attention算子

```python
def apply_prefill_sfa(
    self,
    prefill_preprocess_res: PrefillSFAPreprocessResult,
    attn_metadata: SfaMetadata
):
    if self.cp_size > 1:
        
        output = self.do_cp_balance_attn(
            prefill_preprocess_res,
            attn_metadata
        )
    else:
        output, _, __ = torch_npu.npu_sparse_flash_attention(
            query=prefill_preprocess_res.q_nope,
            key=prefill_preprocess_res.k_nope,
            value=prefill_preprocess_res.k_nope,
            query_rope=prefill_preprocess_res.q_pe,
            key_rope=prefill_preprocess_res.k_pe,
            sparse_indices=prefill_preprocess_res.topk_indices,
            scale_value=self.softmax_scale,
            sparse_block_size=1,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_query=attn_metadata.actual_seq_lengths_query,
            actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_kv,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
        )
    return self.sfa_postprocess(output)

def apply_decode_sfa(
    self,
    prefill_preprocess_res: DecodeSFAPreprocessResult,
    attn_metadata: SfaMetadata
):
    output, _, _ = torch_npu.npu_sparse_flash_attention(
        query=prefill_preprocess_res.q_nope,
        key=prefill_preprocess_res.k_nope,
        value=prefill_preprocess_res.k_nope,
        query_rope=prefill_preprocess_res.q_pe,
        key_rope=prefill_preprocess_res.k_pe,
        sparse_indices=prefill_preprocess_res.topk_indices,
        scale_value=self.softmax_scale,
        sparse_block_size=1,
        block_table=attn_metadata.block_tables,
        actual_seq_lengths_query=attn_metadata.actual_seq_lengths_query,
        actual_seq_lengths_kv=attn_metadata.seq_lens,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
    )
    output = output.squeeze(1)
    return self.sfa_postprocess(output)
```

## vLLM + vLLM-Ascend 方案

> 代码路径：`vllm_ascend/attention/sfa_v1.py`
> 仅实现吸收（MQA）版本，运行在华为昇腾NPU上

### 整体架构

```
AscendSFABackend (AttentionBackend)
  ├── AscendSFAMetadataBuilder (extends MLACommonMetadataBuilder)
  └── AscendSFAImpl (extends MLAAttentionImpl)
        ├── forward()              ← 主入口
        ├── _sfa_preprocess_with_mlapo()  ← MLAPO融合算子路径
        ├── indexer_select_pre_process()  ← Indexer K计算
        ├── indexer_select_post_process() ← Indexer Q + Lightning Indexer
        ├── _execute_sparse_flash_attention_process() ← 稀疏注意力
        ├── _v_up_proj()           ← V解压缩
        └── o_proj()               ← 输出投影
```

### 核心数据流

```
hidden_states
    │
    ├─[MLAPO路径]──────────────────────────────────────────────┐
    │  mla_preprocess() 融合算子:                               │
    │    fused_qkv_a_proj → q_a_layernorm → q_b_proj           │
    │    → split q_nope/q_pe → RoPE → W_UK_T吸收              │
    │    → kv_a_layernorm → npu_kv_rmsnorm_rope_cache          │
    │  输出: ql_nope, q_pe, q_c, k_nope(kv_cache), k_pe(kv_cache)│
    │                                                           │
    ├─[Native路径]──────────────────────────────────────────────┤
    │  fused_qkv_a_proj → split q_c / kv_no_split              │
    │  q_a_layernorm(q_c)                                       │
    │  npu_kv_rmsnorm_rope_cache(kv_no_split) → 写入kv_cache   │
    │  q_b_proj(q_c) → split q_nope/q_pe                       │
    │  bmm(q_nope, W_UK_T) → ql_nope  (吸收: Q@W_UK)          │
    │  npu_interleave_rope(q_pe) → q_pe                         │
    │                                                           │
    ├─[Indexer K]───────────────────────────────────────────────┤
    │  wk(hidden_states) → k_norm → split k_pe/k_nope           │
    │  npu_rotary_mul(k_pe) → cat[k_pe, k_nope] → k_li         │
    │  [可选] Hadamard旋转 → npu_dynamic_quant → int8 k_li      │
    │  npu_scatter_nd_update_(index_cache, k_li)                │
    │                                                           │
    ├─[Indexer Q + Lightning Indexer]───────────────────────────┤
    │  wq_b(q_c) → split q_pe/q_nope                           │
    │  npu_rotary_mul(q_pe) → cat[q_pe, q_nope] → q_li         │
    │  weights_proj(hidden_states) → weights                    │
    │  npu_lightning_indexer(q_li, index_cache, weights)        │
    │    → topk_indices                                         │
    │                                                           │
    ├─[Sparse Flash Attention]──────────────────────────────────┤
    │  npu_sparse_flash_attention(                              │
    │    query=ql_nope, key=kv_cache[0], value=kv_cache[0],    │
    │    query_rope=q_pe, key_rope=kv_cache[1],                │
    │    sparse_indices=topk_indices)                           │
    │    → attn_output                                          │
    │                                                           │
    └─[后处理]──────────────────────────────────────────────────┘
       bmm(attn_output, W_UV) → v_up_proj_result
       o_proj(v_up_proj_result) → output
```

### 关键代码解析

#### 1. MLAPO融合算子路径（Decode优化）

> 条件：`self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS`

MLAPO将Q/K/V的预处理（量化、反量化、LayerNorm、RoPE、KV cache写入、Q吸收投影）融合为单个C++算子`mla_preprocess`，减少NPU kernel launch开销和中间tensor显存占用。

```python
# vllm_ascend/attention/sfa_v1.py :: _sfa_preprocess_with_mlapo
def _sfa_preprocess_with_mlapo(self, hidden_states, kv_cache, cos, sin, slot_mapping, num_input_tokens):
    k_nope, k_pe = kv_cache[0], kv_cache[1]
    ql_nope = torch.empty(...)  # 预分配输出
    q_pe = torch.empty(...)
    q_c = torch.empty(...)      # indexer需要的中间结果

    torch.ops._C_ascend.mla_preprocess(
        hidden_states,
        self.wd_qkv,        # 融合后的QKV权重
        self.deq_scale_qkv, # QKV反量化scale
        self.gamma1, self.beta1,  # q_a_layernorm参数
        self.wu_q,          # q_b_proj权重
        self.qb_deq_scl,    # q_b_proj反量化scale
        self.gamma2,        # kv_a_layernorm参数
        cos, sin,           # RoPE cos/sin
        self.W_UK_T,        # 吸收后的W_UK^T权重
        k_nope, k_pe,       # KV cache (输出)
        slot_mapping,
        ...
        q_out0=ql_nope,     # 输出: 吸收后的q_nope
        kv_cache_out0=k_nope, # 输出: KV cache c_kv
        q_out1=q_pe,        # 输出: q_rope
        kv_cache_out1=k_pe, # 输出: KV cache k_rope
        inner_out=q_c,      # 输出: q_c (给indexer用)
    )
    return hidden_states, ql_nope, q_pe, q_c
```

#### 2. Native路径（Prefill / MLAPO不支持时）

```python
# vllm_ascend/attention/sfa_v1.py :: forward() native分支
qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
q_c, kv_no_split = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
q_c = self.q_a_layernorm(q_c)

# KV: layernorm + RoPE + 写入cache (融合算子)
torch_npu.npu_kv_rmsnorm_rope_cache(
    kv_no_split, self.kv_a_layernorm.weight, cos, sin,
    slot_mapping, kv_cache[1], kv_cache[0], ...)

# Q: q_b_proj → split nope/rope → 吸收投影 + RoPE
ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
q_pe = self.rope_single(q_pe, cos, sin)
```

其中`_q_proj_and_k_up_proj`实现了**吸收**（MQA模式）的关键步骤：

```python
# vllm_ascend/attention/sfa_v1.py :: _q_proj_and_k_up_proj
def _q_proj_and_k_up_proj(self, x):
    q_nope, q_pe = self.q_proj(x)[0].view(-1, self.local_num_heads, self.qk_head_dim) \
        .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    # 吸收: q_nope @ W_UK^T → ql_nope (将Q从head_dim映射到kv_lora_rank)
    q_nope = q_nope.transpose(0, 1)          # (N, B, P)
    ql_nope = torch.bmm(q_nope, self.W_UK_T) # (N, B, P) x (N, P, L) → (N, B, L)
    return ql_nope.transpose(0, 1), q_pe      # (B, N, L), (B, N, d_rope)
```

#### 3. Indexer K计算与存储

```python
# vllm_ascend/attention/sfa_v1.py :: indexer_select_pre_process
def indexer_select_pre_process(self, x, cos, sin):
    k_li, _ = self.wk(x)                # [b*s, 7168] @ [7168, 128] = [b*s, 128]
    k_li = self.k_norm(k_li).unsqueeze(1) # LayerNorm
    k_li = k_li.view(-1, 1, self.head_dim)

    # RoPE: split rope/nope → 旋转rope部分 → cat回来
    k_li_pe, k_li_nope = torch.split(k_li, [self.qk_rope_head_dim, ...], dim=-1)
    k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
    k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)

    # [可选] Sparse C8: Hadamard旋转 + 动态量化到int8
    if self.use_sparse_c8_indexer:
        k_li = k_li @ AscendSFAImpl.k_hadamard
        k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li, dst_type=torch.int8)

    return k_li, k_li_scale
```

#### 4. Indexer Q + Lightning Indexer

```python
# vllm_ascend/attention/sfa_v1.py :: indexer_select_post_process
def indexer_select_post_process(self, x, q_c, kv_cache, attn_metadata, cos, sin, ...):
    weights, _ = self.weights_proj(x)     # 标量权重 w^I
    q_li, _ = self.wq_b(q_c)             # Indexer Q投影
    q_li = q_li.view(-1, self.n_head, self.head_dim)

    # RoPE (同K的处理)
    q_li_pe, q_li_nope = torch.split(q_li, [self.qk_rope_head_dim, ...], dim=-1)
    q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
    q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

    # Lightning Indexer: 计算索引分数 + TopK选择 (融合算子)
    if self.use_sparse_c8_indexer:
        # int8量化路径: npu_lightning_indexer_quant
        topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
            query=q_li, key=kv_cache[2], weights=weights,
            query_dequant_scale=q_li_scale, key_dequant_scale=kv_cache[3], ...)
    else:
        # bf16路径
        topk_indices = torch.ops._C_ascend.npu_lightning_indexer(
            query=q_li, key=kv_cache[2], weights=weights,
            actual_seq_lengths_query=..., actual_seq_lengths_key=...,
            block_table=attn_metadata.block_table,
            layout_query="TND", layout_key="PA_BSND",
            sparse_count=2048, sparse_mode=3)
    return topk_indices
```

#### 5. Sparse Flash Attention

```python
# vllm_ascend/attention/sfa_v1.py :: _execute_sparse_flash_attention_process
def _execute_sparse_flash_attention_process(self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, ...):
    attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope,           # 吸收后的Q (B, N, kv_lora_rank)
        key=kv_cache[0],         # c_kv cache
        value=kv_cache[0],       # value = key (MQA吸收模式)
        sparse_indices=topk_indices,
        scale_value=self.scale,
        sparse_block_size=1,
        block_table=block_table,
        query_rope=q_pe,         # Q的RoPE部分
        key_rope=kv_cache[1],    # K的RoPE cache
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3)
    return attn_output
```

#### 6. V解压缩 + 输出投影

```python
# 吸收模式下，attn_output的维度是(B, N, kv_lora_rank)
# 需要通过W_UV投影回v_head_dim
attn_output = self._v_up_proj(attn_output)  # bmm(attn_output, W_UV)
output[...] = self.o_proj(attn_output)[0]
```

## SGLang 方案

> 代码路径：`sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py` (Indexer)
> `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` (Sparse Attention)
> `sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` (DSA Prepare/Core)
> 仅实现吸收（MQA）版本，运行在华为昇腾NPU上

### 路由与分发

SGLang 将 DeepSeek V3.2 / GLM5 的注意力分为三条路径，通过 `AttentionBackendRegistry` 自动选择：

| 方法 | 条件 | 使用场景 |
|------|------|----------|
| `MHA_NPU` | Ascend后端 + 无 indexer | 非NSA模型的Prefill |
| `MLA_NPU` | Ascend后端 + 无 indexer | 非NSA模型的Decode |
| **`DSA_NPU`** | Ascend后端 + 有 indexer | NSA模型 Prefill+Decode 统一入口 |

```python
# sglang/.../attention_backend_handler.py:38-53
def handle_attention_ascend(attn, forward_batch):
    # 关键判断：模型是否有 indexer 决定了走 DSA 还是普通 MLA
    if hasattr(attn, "indexer"):
        return AttnForwardMethod.DSA_NPU    # DeepSeekV3.2 / GLM5 → SFA路径
    else:
        return AttnForwardMethod.MHA_NPU if is_prefill else AttnForwardMethod.MLA_NPU
```

与 vLLM-Ascend 的 `AscendSFAImpl extends MLAAttentionImpl` 不同，SGLang 不做继承，而是独立出 `DSA_NPU` 路径，prepare 和 core 分阶段执行。

### 整体架构

```
DeepseekV2AttentionMLA
  ├── Indexer (MultiPlatformOp)          ← 跨平台 Indexer 抽象
  │     ├── forward_npu()                ← NPU 路径
  │     └── forward_cuda()               ← CUDA/HIP 路径 (FP8量化 + deep_gemm)
  │
  └── Dispatch (via AttentionBackendRegistry)
        ├── forward_dsa_prepare_npu()    ← QKV投影 + 吸收 + Indexer
        │     └── forward_dsa_core_npu() ← Sparse Attention + V解压缩 + O投影
        ├── forward_mla_prepare_npu()    ← 非NSA MLA路径
        └── forward_mha_prepare_npu()    ← 非NSA MHA路径
```

### 核心数据流

```
hidden_states
    │
    ├─[MLA投影]──────────────────────────────────────────┐
    │  fused_qkv_a_proj_with_mqa → split q_lora / kv_no_split │
    │                                                     │
    ├─[Q路径]────────────────────────────────────────────┤
    │  q_a_layernorm(q_lora)                              │
    │  q_b_proj(q) → reshape [tokens, n_heads, qk_head_dim]│
    │  split q_nope / q_pe                                │
    │  bmm(q_nope.T, W_UK^T).T → q_nope_out (吸收: Q@W_UK)│
    │                                                     │
    ├─[KV路径]───────────────────────────────────────────┤
    │  kv_a_layernorm(kv_no_split[..., :kv_lora_rank])    │
    │  k_pe = kv_no_split[..., kv_lora_rank:]             │
    │  RoPE(q_pe), RoPE(k_pe)                             │
    │                                                     │
    ├─[Indexer Q/K投影]──────────────────────────────────┤
    │  wq_b(q_lora) → reshape [bs, H_I, d_I]              │
    │  split q_pe/q_nope → npu_rotary_mul(q_pe) → cat    │
    │                                                      │
    │  wk(hidden_states) → k_norm → split k_pe/k_nope     │
    │  npu_rotary_mul(k_pe) → unsqueeze(1) → cat          │
    │  写入 index_k_cache (bf16, NPU路径不量化!)          │
    │                                                      │
    ├─[Indexer Weights]───────────────────────────────────┤
    │  weights_proj(hidden_states) → [bs, H_I] 标量权重   │
    │  [可选] 多流并行：weights与Q/K投影在不同stream上重叠│
    │                                                      │
    ├─[Lightning Indexer]─────────────────────────────────┤
    │  npu_lightning_indexer(q_li, key_cache, weights,    │
    │    sparse_count=2048, sparse_mode=3)                 │
    │  → topk_indices [bs, topk]                           │
    │                                                      │
    ├─[Sparse Flash Attention]────────────────────────────┤
    │  npu_sparse_flash_attention(                         │
    │    query=q_nope_out (吸收后, kv_lora_rank维),        │
    │    key=k_nope, value=k_nope,  (MQA吸收模式)          │
    │    query_rope=q_pe, key_rope=k_pe,                  │
    │    sparse_indices=topk_indices,                      │
    │    sparse_mode=3, attention_mode=2)                  │
    │  → attn_output [tokens, n_heads, kv_lora_rank]      │
    │                                                      │
    └─[后处理]───────────────────────────────────────────┘
       bmm(attn_output, W_UV) → [tokens, n_heads, v_head_dim]
       o_proj → output [tokens, hidden_size]
```

### 关键代码解析

#### 1. Q/KV 投影与吸收 (MLA Prepare)

> 对应公式：`c_t^Q = W_{DQ} h_t` → `W_{UQ} c_t^Q` → split `q_{t,i}^C` / `q_{t,i}^R`

```python
# deepseek_v2_attention_mla_npu.py :: forward_dsa_prepare_npu (Native路径, 非MLAPO)
fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
# 对应公式: W_{DQ} h_t → c_t^Q,  W_{DKV} h_t → c_t^{KV},  W_{KR} h_t → k_t^R
q, latent_cache = fused_qkv_a_proj_out.split(
    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
)
q_lora = m.q_a_layernorm(q)                           # layernorm on c_t^Q
k_nope, k_pe = latent_cache.unsqueeze(1).split(        # 拆分 KV 潜向量
    [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
)
k_nope = m.kv_a_layernorm(k_nope)                     # layernorm on c_t^{KV}
q = m.q_b_proj(q_lora)[0]                              # W_{UQ} c_t^Q → [tokens, n_heads, qk_head_dim]
q = q.view(-1, m.num_local_heads, m.qk_head_dim)

q_nope, q_pe = q.split(                                # 拆分 Content / Rotary
    [m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1
)  # q_nope → q_{t,i}^C,  q_pe → q_{t,i}^R (RoPE前)

# 吸收: q_nope @ W_UK^T → 将 Q 从 head_dim 映射到 kv_lora_rank
# 对应 MQA 吸收模式: q_{t,i}^{C\top} W_{UK}^\top 提前计算, 使得 Attention 中
# QK^\top = (q_nope @ W_UK^T) @ c_{KV}^\top 等效于原始 q_nope @ (W_UK @ c_{KV})^\top
q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)
q_nope_out = q_nope_out.transpose(0, 1)               # [tokens, n_heads, kv_lora_rank]
```

#### 2. Indexer Q/K 投影与 RoPE

> 对应公式: `q_{t,j}^I` 源自 `h_t`, `k_s^I` 源自 `h_s`
> 注意 Indexer K 是**单头的**（`unsqueeze(1)`），即 $H_I$ 头共享同一个 $k_s^I$

```python
# nsa_indexer.py :: forward_npu (neox_style 路径, DeepSeekV3.2使用)
# ============ Indexer Q: q_{t,j}^I 计算 ============
# wq_b 对应 q^I 投影, 输入是 q_lora (即 c_t^Q) 而非 hidden_states
q = self.wq_b(q_lora)[0]                              # q_lora: [bs, 1536], q: [bs, 64*128]
q = q.view(bs, self.n_heads, self.head_dim)           # [bs, H_I=64, d_I=128]
q_pe, q_nope = torch.split(                            # split RoPE / nope
    q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
)  # q_pe: [bs, 64, 64], q_nope: [bs, 64, 64]

q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin)       # RoPE on q_{t,j}^{I,R}
q_pe = q_pe.view(bs, self.n_heads, self.rope_head_dim)
q = torch.cat([q_pe, q_nope], dim=-1)                 # q_{t,j}^I = [q_{t,j}^{I,R}; q_{t,j}^{I,C}]

# ============ Indexer K: k_s^I 计算 ============
# wk 对应 k^I 投影, 输入是 hidden_states (即 h_s)
k_proj = self.wk(x)[0]                                # hidden_states: [bs, 7168] → [bs, 128]
k = self.k_norm(k_proj)                               # LayerNorm on k_s^I
k_pe, k_nope = torch.split(                            # split RoPE / nope
    k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
)

k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin)   # RoPE on k_s^{I,R}
k_pe = k_pe.view(bs, 1, self.rope_head_dim)           # ★ 单头: [bs, 1, 64]
k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)    # k_s^I: [bs, 1, 128]

# 存入 Index K Cache (bf16, NPU路径不量化)
# CUDA/HIP路径会走 Hadamard旋转 + FP8量化, 但 NPU 上直接存 bf16
forward_batch.token_to_kv_pool.set_index_k_buffer(
    layer_id, forward_batch.out_cache_loc, k
)
```

#### 3. Indexer Weights 计算 (多流并行)

> 对应公式: $w_{t,j}^I$ 源自 $h_t$

```python
# nsa_indexer.py :: forward_npu
# weights_proj 对应 w^I 投影
# [可选] 通过多流将 weights 计算与 Q/K 投影重叠, 隐藏延迟
if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
    indexer_weight_stream = get_indexer_weight_stream()
    indexer_weight_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(indexer_weight_stream):
        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x.float())[0].to(torch.bfloat16)  # [bs, H_I]
        weights.record_stream(indexer_weight_stream)
        weights_event = indexer_weight_stream.record_event()
else:
    x = x.view(-1, self.hidden_size)
    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)      # [bs, H_I]
```

#### 4. Lightning Indexer 调用

> 对应公式:
> $$I_{t,s} = \sum_{j=1}^{H_I} w_{t,j}^I \cdot \text{ReLU}(q_{t,j}^I \cdot k_s^I)$$
> $$\mathcal{S}_t = \{ s \mid I_{t,s} \in \text{Top-k}(I_{t,:}) \}$$

```python
# nsa_indexer.py :: forward_npu
# 从 cache 中读取所有历史 token 的 Indexer K
past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

topk_indices = torch_npu.npu_lightning_indexer(
    query=q.view(-1, self.n_heads, self.head_dim),   # q_{t,j}^I: [tokens, H_I=64, d_I=128]
    key=past_key_states,                               # k_s^I:   [total_tokens, 1, 128]
    weights=weights,                                   # w_{t,j}^I: [tokens, H_I=64]
    actual_seq_lengths_query=actual_seq_lengths_q,
    actual_seq_lengths_key=actual_seq_lengths_kv,
    block_table=block_table,
    layout_query="TND",                                # tokens × n_heads × dim
    layout_key="PA_BSND",                              # Paged Attention, B×S×N×D
    sparse_count=self.index_topk,                      # k = 2048 (Top-k 选择)
    sparse_mode=3,                                     # SFA模式
)
# topk_indices: [tokens, index_topk] — 每个 token 选中的历史 token 下标
# 对应公式中的 S_t 集合
return topk_indices[0]
```

#### 5. Sparse Flash Attention

> 对应公式:
> $$o_{t,i} = \sum_{j \in \mathcal{S}_t} \text{Softmax}_j(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}}) v_{j,i}^C$$
>
> 吸收模式下 `query=q_nope_out` (已乘 $W_{UK}^\top$), `key=value=k_nope` ($c_t^{KV}$ 潜向量)

```python
# ascend_backend.py :: forward_sparse
def forward_sparse(self, q, k, v, layer, forward_batch,
                   q_rope=None, k_rope=None, topk_indices=None):
    # 保存 KV cache
    if save_kv_cache:
        k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
        k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, k_rope
        )

    q_nope, q_pe = q, q_rope                        # Q: 已吸收 [tokens, n_heads, kv_lora_rank]
    k_nope, k_pe = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

    # 调用 NPU 稀疏注意力算子
    attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
        query=q_nope,                                # 吸收后 Q: [tokens, n_heads, kv_lora_rank]
        key=k_nope,                                  # c_t^{KV} cache
        value=k_nope,                                # MQA模式: value == key (吸收)
        query_rope=q_pe,                             # Q RoPE: [tokens, n_heads, rope_dim]
        key_rope=k_pe,                               # K RoPE cache
        sparse_indices=topk_indices,                 # S_t: 稀疏选择的 token 下标
        scale_value=layer.scaling,                   # 1/√(d_h + d_h^R)
        block_table=self.forward_metadata.block_tables,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,                            # Sparse MLA
    )
    # attn_out: [tokens, n_heads, kv_lora_rank] — 对应 o_{t,i} (吸收模式)
    return attn_out
```

#### 6. V解压缩 + 输出投影 + 层间 TopK 复用

> V解压缩对应: $v_t^C = W_{UV} c_t^{KV}$

```python
# deepseek_v2_attention_mla_npu.py :: forward_dsa_core_npu
attn_output = m.attn_mqa(
    q_nope_out.contiguous(),                         # 吸收后 Q
    k_nope.contiguous(),                             # c_t^{KV}
    k_nope.contiguous(),                             # value = key (MQA)
    forward_batch, save_kv_cache=True,
    q_rope=q_pe.contiguous(), k_rope=k_pe.contiguous(),
    topk_indices=topk_indices,
)
attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)
# attn_output: [tokens, n_heads, kv_lora_rank]

# V解压缩: bmm(attn_output, W_UV) → [tokens, n_heads, v_head_dim]
attn_bmm_output = torch.empty(
    (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
    dtype=attn_output.dtype, device=attn_output.device,
)
attn_output = attn_output.contiguous()
torch.ops.npu.batch_matmul_transpose(                # NPU 加速 batched matmul
    attn_output, m.w_vc, attn_bmm_output
)
# 输出投影: W_O
attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
output, _ = m.o_proj(attn_bmm_output)

# ============ 层间 TopK 复用 ============
# 根据 index_topk_freq / index_topk_pattern 控制哪些层跳过 Indexer 计算
# 参考论文: https://arxiv.org/abs/2603.12201
if not m.next_skip_topk:
    return output, None        # 下层不需要复用 topk_indices
else:
    return output, topk_indices # 传递给下层复用
```

**层间复用配置** (`deepseek_v2.py:1263-1275`):

```python
# 两种模式控制层间 topk 复用:
self.index_topk_freq = getattr(config, "index_topk_freq", 1)
# index_topk_freq=1: 每层都算 Indexer
# index_topk_freq=2: 每2层算一次, 奇数层复用上层结果

self.index_topk_pattern = getattr(config, "index_topk_pattern", None)
# pattern="ABAB": A=计算, B=跳过, 如 ["A","B","A","B","A","B"...]
if self.index_topk_pattern is None:
    self.skip_topk = max(layer_id - 1, 0) % self.index_topk_freq != 0
    self.next_skip_topk = layer_id % self.index_topk_freq != 0
else:
    self.skip_topk = self.index_topk_pattern[layer_id] == "S"      # Skip
    self.next_skip_topk = self.index_topk_pattern[layer_id + 1] == "S"
```

### 与 MindIE / vLLM-Ascend 关键差异

| 维度 | MindIE | vLLM-Ascend | SGLang |
|------|--------|-------------|--------|
| Indexer 抽象 | inline 在 sparse_attention.py | 内嵌在 AscendSFAImpl | **独立 Indexer 类 (MultiPlatformOp)** |
| Index K Cache | bf16 (不量化) | bf16 (不量化) | **bf16 (不量化)** |
| CUDA Index Cache | N/A | N/A | **FP8 量化 (Hadamard + dynamic_quant)** |
| RoPE 算子 (Indexer) | `npu_interleave_rope` | `npu_rotary_mul` | **`npu_rotary_mul`** |
| 多流策略 | 单流 | 单流融合算子 (MLAPO) | **多流重叠 weights, 也有MLAPO** |
| Skip-TopK | 无 | 无 | **支持 freq/pattern 层间复用** |
| 路由方式 | 直接调用 | 继承体系 (extends MLAAttentionImpl) | **Backend Registry → DSA_NPU 独立路径** |
| CP支持 | Indexer + Attn 各自CP | Indexer + Attn 各自CP | **Indexer + Attn 各自有 CP balance** |
| Attention 调用 | `npu_sparse_flash_attention` | `npu_sparse_flash_attention` | `npu_sparse_flash_attention` |

**三个项目在 NPU 上的共识：**
- Indexer K Cache 都存 bf16，不做量化（与 CUDA 路径的 FP8 量化不同）
- Indexer K 都是单头（`unsqueeze(1)`），对应公式中 $k_s^I$ 为所有 $H_I$ 头共享
- 稀疏注意力都仅实现 MQA（吸收）版本：`key=value=c_kv`, `query=absorbed_q`
- 底层都调用同一个 NPU 算子：`npu_sparse_flash_attention(sparse_mode=3, attention_mode=2)`
