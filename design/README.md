# design —— Tokenizer 设计文档

本目录是自研 tokenizer 的**设计真相源**：机制 rationale、契约边界、定案结论都写在这里，
代码只负责实现。改动 pre-tokenizer / post-processor / model / 训练路径时，先改这里，再改实现。

阅读顺序：先 `tokenizer-architecture.md` 建立当前边界认知，再按主题进入对应文档。

## 1. 基础：当前架构与边界

| 文档 | 内容 |
|---|---|
| [`tokenizer-architecture.md`](tokenizer-architecture.md) | crate 分层（`tk-encode` / `tk-serialize` / `tk-convert` / `tk-train`）、pipeline 数据流、encode/decode 的运行时边界 |
| [`pre-tokenizer-modules.md`](pre-tokenizer-modules.md) | 12 个配置名 → 10 个 runtime 变体的映射；`ByteLevel` / `Metaspace` 的降级路径；delimiter behavior 语义 |
| [`postprocessor-scheme.md`](postprocessor-scheme.md) | 当前只有 `PipelinePostProcessor { single, pair }` 两个 `Template`；5 种历史配置形态如何被 `tk-convert` 降级 |

## 2. 定案：自研最小路径

| 文档 | 内容 |
|---|---|
| [`minimal-tokenizer-path.md`](minimal-tokenizer-path.md) | 不改仓库源码即可运行的最小方案：`SPECIALS` / `PROTOCOL_TOKENS` / `initial_alphabet` / `BYTE_TOKENS` 四层词表、固定管线、自训脚本与门禁 |
| [`chat-templates.md`](chat-templates.md) | Qwen3.8（Jinja）与 DeepSeek-V4.1（Python encoder）两套聊天协议；绑定 commit + SHA-256，不用浮动 `main`；工具调用 / thinking / 多模态占位符 |
| [`uax29-sentence-pretokenizer.md`](uax29-sentence-pretokenizer.md) | **最终预分词定案**：C2 `CLASS_REGEX` + `Split(behavior="merged_with_next", invert=false)`；两条不可改的约束与正则全文 |
| [`pretok-performance.md`](pretok-performance.md) | C2/C3 吞吐数据与模型层（BPE / WordPiece / Unigram）基准，C3 只作参照 |

## 3. 调研

| 文档 | 内容 |
|---|---|
| [`tokenizer-landscape-survey.md`](tokenizer-landscape-survey.md) | Hub Top-20k 全量调查（数据快照 2026-09-22）：12,171 根 / 6,207 家族；`normalizer=None`、字符类切分、BPE+byte_fallback 的普及率与排除线 |

## 约定

- 当前 Rust core 是**只读推理管线**（`PipelineTokenizer` + `tk-serialize` reader）；setters、`save`、
  trainers、完整 original-offset 回填都不在契约内，缺失项见根目录 `REQUIRED_FOR_V1.md`。
  不要按旧版 `Tokenizer::new` / `add_tokens` 写法补文档或代码。
- 引用配置形态时区分「历史配置形态」与「当前 runtime 变体」；canonical reader 只接受 2.0 写法。
- 性能结论必须带 CPU/OS 与轮次；C1 `sentence_breaks` 是历史字段，当前 `Split` 不再序列化它。
