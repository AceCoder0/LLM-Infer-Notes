# NPU 推理框架对比分析 - 索引

## 📁 文档结构

```
NPU_Framework_Analysis/
├── README.md                              # 本索引文件
├── NPU_Framework_Comparison_Report.md     # 总体对比报告
└── NPU_Framework_Detailed_Comparison.md   # 详细逐项对比
```

## 📄 文档说明

### 1. NPU_Framework_Comparison_Report.md (总体对比报告)

**内容概要**：
- 框架概述与定位
- 架构设计对比（分层架构图）
- Attention 机制实现对比
- KV Cache 管理对比
- 调度器设计对比
- 分布式与并行策略
- 量化和压缩支持
- 推测解码支持
- 接口与易用性
- 总结对比表

**适合阅读**：希望快速了解三个框架整体差异的读者

---

### 2. NPU_Framework_Detailed_Comparison.md (详细逐项对比)

**内容概要**：
- Attention 核心接口详细代码对比
- MLA 实现细节对比
- 内存管理和 KV Cache 实现对比
- 分页分配器代码对比
- 调度器算法对比
- 分布式通信实现对比
- 量化技术实现对比
- 推测解码实现对比
- 模型适配和扩展性对比
- 关键文件索引

**适合阅读**：需要深入理解具体实现细节的开发者

---

## 🔍 快速导航

### 按主题查找

| 主题 | 总体报告章节 | 详细对比章节 |
|------|-------------|-------------|
| 架构设计 | 第2章 | - |
| Attention | 第3章 | 第1章 |
| KV Cache | 第4章 | 第2章 |
| 调度器 | 第5章 | 第3章 |
| 分布式通信 | 第6章 | 第4章 |
| 量化 | 第7章 | 第5章 |
| 推测解码 | 第8章 | 第6章 |
| 接口使用 | 第9章 | - |

---

## 📊 快速对比表

| 维度 | SGLang NPU | vLLM-Ascend | MindIE-LLM |
|------|------------|-------------|------------|
| **架构风格** | Python-first | 插件化 | C++核心 |
| **核心语言** | Python | Python+C++ | C+++Python |
| **学习曲线** | 中等 | 较高 | 较高 |
| **主要优势** | RadixAttention | 插件生态 | 性能极致 |
| **推荐场景** | 灵活定制 | 通用部署 | 企业级 |

---

## 🔗 相关资源

### SGLang NPU
- GitHub: https://github.com/sgl-project/sglang
- 文档: https://docs.sglang.io/
- NPU支持: https://github.com/sgl-project/sglang/tree/main/docs_new/docs/hardware-platforms/ascend-npus

### vLLM-Ascend
- GitHub: https://github.com/vllm-project/vllm-ascend
- 文档: https://docs.vllm.ai/projects/ascend/en/latest/

### MindIE-LLM
- 代码仓: (昇腾官方)
- 文档: https://www.hiascend.com/software/mindie

---

## 📝 报告信息

- **生成日期**: 2026-04-29
- **分析版本**:
  - SGLang: latest
  - vLLM-Ascend: v0.13.0
  - MindIE-LLM: 2.3.0
- **基于源码**: 各仓库最新 main/dev 分支

---

*如需补充或修正，请提交 Issue 或 PR*
