# Tokenizer 当前架构与边界

> 面向自研 tokenizer 的实现摘要。本文只描述当前 checkout：主线是文本如何经过只读推理管线成为 ID 序列，以及当前 `Encoding` 实际包含什么；训练和完整 original-offset 输出不在当前 core 保证内。

## 1. 当前分层

| Crate | 当前职责 |
|---|---|
| `tokenizers/tk-encode/` | 只读推理：span-based `PipelineTokenizer`、组件实现、四种 model、encode/decode |
| `tokenizers/tk-serialize/` | canonical `tokenizer.json` 2.0 reader，以及 `serialize` feature 下的 writer |
| `tokenizers/tk-convert/` | 旧配置到 canonical 2.0 的 JSON→JSON 升级，不构造运行时对象 |
| `tokenizers/tk-train/` | trainer 源码仍在树内，但被 workspace 排除，当前 core 与 bindings 均不可用 |

`PipelineTokenizer` 主要公开 `from_parts`、getters、padding/truncation 解析、`encode`/`encode_into`、`decode`/`decode_stream`。可变 `Tokenizer` authoring object、组件 setters、`add_tokens`、文件式 `save`、trainers 和完整 original-offset 回填尚未迁入当前 pipeline。`tk-serialize::to_json` 能序列化现成 pipeline，但这不等于恢复了可变对象 API。

运行时边界：

| 边界 | 当前契约 |
|---|---|
| `Normalizer` | `(&str, byte offset) -> Cow<str>`；不返回 alignment |
| `PreTokenizer` | 输出当前文本 chunk 内的 UTF-8 byte `Span` |
| `Model` | 消费 span，只追加 `PipelineToken(u32)`；ID 反查由 `PipelineModel` 提供 |
| `PipelinePostProcessor` | 运行时只有 single/pair 两个 `Template`，写入 ID 与 `type_id` |
| `Decoder` | token 字符串序列经 decoder chain 还原文本 |

## 2. Pipeline 数据流

### 2.1 编码

```text
Input::Single / Pair
  -> 两段各自从 offset 0 编码
  -> raw AddedVocabulary：先切出匹配的 added/special token
  -> 普通文本 chunk 执行 NormalizerChain
  -> normalized AddedVocabulary：再切出 normalized added token
  -> PreTokenizer 输出 Span
  -> Model 对每个 Span 追加 ID
  -> truncate；启用自动 special 时先预留模板 token 数
  -> single/pair Template 写入 special ID 与 type_id
  -> EncodeHandle::wait 按需 padding
  -> pipeline::Encoding
```

关键语义：

- `EncodeOptions::default()` 会添加 post-processor special token；`encode_special_tokens=false` 会把输入中匹配的 special token 直接替换成 added ID。
- `add_special_tokens` 只控制 post-processor 包装；`encode_special_tokens` 控制输入内 special token 是否进入 model，两者不能混为一谈。
- pair 的 B 段也从 0 编码；默认模板给 A/B 写 `0/1`，自定义 canonical template 可指定其他 `type_id`。
- 当前 `truncate_pair` 只保留首个窗口，不生成 `overflowing`，也不使用 `stride`。`EncodeHandle::wait` 后的 padding 只扩展 ID、`type_id` 和 `attention_mask`。

### 2.2 解码

```text
IDs
  -> AddedVocabulary / Model::id_to_token
  -> 可选过滤 special token
  -> Decoder chain；未配置 decoder 时用空格连接
```

byte-level BPE 在加载时已把词表转成原始 bytes，解码走直接拼接路径。`PipelineDecodeStream` 保留必要前缀，等待多 byte UTF-8 或跨 token decoder 状态稳定后再输出。

## 3. Span、offset 与 `Encoding` 边界

`Span` 只表示当前 normalized chunk 内的 byte range。`PreTokenizer` 必须保证 range 有序、未越界且两端位于 UTF-8 字符边界；model 随后用 unchecked slice 消费它。`Normalizer` 只返回 `Cow<str>`，没有 normalized→original alignment，因此当前 span 不能直接充当 original offset。

| 类型或 binding | 当前实际输出 |
|---|---|
| `pipeline::Encoding` | `PipelineToken` ID 列表；可选 `type_ids`、可选 `attention_mask` |
| `tokenizer::Encoding` | 独立完整值类型，含 tokens、words、offsets、`special_tokens_mask`、overflowing、sequence ranges 等字段 |
| Python `Encoding` | `ids`、`type_ids`、`attention_mask`，并提供只读 NumPy view |
| Node binding | 只返回 `Uint32Array` ID，不暴露 `Encoding` 对象 |

不要把这四层混成同一保证：

- `PipelineTokenizer::encode` 返回第一种稀疏类型；Python binding 将缺失的 `type_id`/`attention_mask` 默认物化为全 `0`/全 `1`。
- 完整 `tokenizer::Encoding` 的 `truncate`、`merge`、`overflowing` helper 仍存在，但当前 pipeline 不调用它们；其完整 offset 字段也不是 encode 输出。
- 当前 `pipeline::Encoding` 与 Node surface 均无 token 文本、word ID、byte/char offset、special-token mask 或 overflowing 输出。Python 可通过 `tokenize`/`decode_tokens` 另取 token 字符串，但其 `Encoding` 仍没有上述映射字段。

`AddedToken` 保留 `content`，以及 `single_word`、`lstrip`、`rstrip`、`normalized`、`special` 五个行为字段。raw 与 normalized matcher 分开；默认先切 added token，只有 `encode_special_tokens=true` 时 special token 被留给 model，普通 added token 仍会被切出。

## 4. Model 对照

| Model | 当前编码机制 | canonical 字段与边界 |
|---|---|---|
| BPE | byte/character atom 按 merge rank 合并；完整词表项可走 fold probe；每线程 scratch 带 word cache | `vocab`、有序 `merges`、`byte_level`、unk/prefix/suffix、`fuse_unk`、`byte_fallback`、`ignore_merges`；`dropout > 0` 当前拒绝 |
| WordPiece | 对每个 pre-token 做贪心最长匹配；续接片段加 prefix；任一片失败或超长则整词回退 UNK | `unk_token`、`continuing_subword_prefix`、`max_input_chars_per_word`、`vocab` |
| WordLevel | 整个 pre-token 精确查表；不拆子词 | `unk_token`、`vocab`；缺少可解析的 UNK 时报错 |
| Unigram | Trie 建候选，lattice Viterbi 取最高分路径；未知字符可走 byte fallback | `vocab` 数组索引即 ID，另有 `unk_id`、`byte_fallback`；canonical load 使用确定性 Viterbi |

四种 model 当前都只向 pipeline 追加 ID。writer 按 ID 排序普通 vocab；BPE merge 数组索引就是 rank，Unigram vocab 数组索引就是 ID。训练统计、EM、剪枝和 trainer 构建不属于这条运行时路径。

## 5. 其他组件边界

- Normalizer 支持 `Sequence`、`MetaspaceNormalizer`、`Replace`、`Prepend`、`Strip`、`Lowercase`、`ByteLevel`、`BertNormalizer`、`StripAccents`、`NFC`/`NFD`/`NFKC`/`NFKD`、`Nmt`、`Precompiled`；reader 会把 `Sequence` 展平为顺序链，无改写时保持借用。
- canonical pre-tokenizer 包含 `Sequence`、`BertPreTokenizer`、`CharDelimiterSplit`、`Digits`、`FixedLength`、`Punctuation`、`Split`、`UnicodeScripts`、`Whitespace`、`WhitespaceSplit`。legacy `Metaspace` 与 `ByteLevel` pre-tokenizer 必须先由 `tk-convert` 拆成 normalizer/model flag 加 `Split`。
- canonical post-processor 只有 `TemplateProcessing`。legacy `ByteLevel`、`BertProcessing`、`RobertaProcessing` 及可无歧义转换的 `Sequence` 由 `tk-convert` 降为 single/pair template；无法唯一确定的情况拒绝转换。
- Decoder 支持 `BPEDecoder`、`ByteLevel`、`WordPiece`、`Metaspace`、`CTC`、`Fuse`、`Strip`、`ByteFallback`、`Replace`、`Sequence`；canonical 组件表见 [SPEC.md](../tokenizers/tk-serialize/SPEC.md)。

## 6. 序列化与 bindings

canonical 2.0 顶层键如下，writer 固定按此顺序输出：

```text
version -> truncation -> padding -> role_to_token -> added_tokens
        -> normalizer -> pre_tokenizer -> post_processor -> decoder -> model
```

- `tk-serialize::from_json` / `from_json_file` 只读 2.0，直接构造 `PipelineTokenizer`；其他版本、legacy 无 `type` model、旧 merge 字符串、文件型 vocab 等会被拒绝。
- `tk-serialize::to_json` 在 `serialize` feature 下按稳定顺序输出。reader 会先按 ID 排序 added tokens，再针对具体 model 注册，避免复用 model ID 时静默改号。
- bindings 的 `from_file` 会先调用 `tk-convert::canonicalize_file`，所以它们能兼容磁盘上的 1.0 文件；这不改变 `tk-serialize` 自身只接受 canonical 2.0 的边界。

Python binding 使用 PyO3 + maturin，导出持有只读 pipeline 的 `Tokenizer`、稀疏 `Encoding`、`Padding`、`Truncation`。它支持单文本 `encode`/`encode_batch`、`tokenize`、`decode`/`decode_tokens`，以及经 `huggingface_hub` 下载后的 `from_pretrained`；padding/truncation 可通过 mutex 属性修改或按调用覆盖。encode/decode 重操作释放 GIL；repr/pickle 用 `to_json` 保存 pipeline 部分及 options；没有 trainers、pair 输入、offset API 或公开 `save`。

Node binding 使用 NAPI-RS，当前公开面只有 `PipelineTokenizer.from_file`、`encode`、`encode_bytes_into`。`encode` 返回新 `Uint32Array`；`encode_bytes_into` 写入调用方 buffer，避免 JS string→UTF-8 与结果 ArrayBuffer 分配。两端都没有旧组件 wrapper、trainer 或 async task。修改 Rust 后必须重建 native extension；Python stub/style 检查还会重生成 stub。

## 7. 性能关键点

- PreTokenizer 复用 `Vec<Span>`，normalizer 无改写时返回 `Cow::Borrowed`，不为每个 split 分配 owned string。
- `ScratchPool` 分片复用 tags、bitmaps、span buffers、model scratch 和跨调用 word cache。
- 启用 `parallelism` 时，总输入达到 8 KiB 且能拆出多个 task 才进入并行计划；Rayon 线程数由 `RAYON_RS_NUM_THREADS` 控制。
- BPE cache、fold probe 与 scratch 共同服务重复 pre-token；当前不接受正 dropout，因此不存在“随机 dropout 路径”保证。
- `encode_into` 可直接向调用方 ID buffer 追加；Node `encode_bytes_into` 再省一层 JS 边界分配。

## 8. 自研训练路径

当前 core 不能训练，也没有完整 original-offset 输出。可执行原型固定使用 PyPI `tokenizers==0.23.2`，避免与只读 runtime 混淆；完整脚本见 [minimal-tokenizer-path.md](./minimal-tokenizer-path.md)。

1. 用字符集合作为 BPE initial alphabet，训练 `BPE(byte_fallback=True, unk_token="<unk>")`。
2. 训练后补齐完整 256 个 `<0xHH>` vocab 项；`byte_fallback` 不会自动生成这些 ID。
3. `normalizer=None`，预分词采用 [uax29-sentence-pretokenizer.md](./uax29-sentence-pretokenizer.md) 的字符类 `Split(..., merged_with_next)`。
4. `post_processor=None`，先手工控制 special/protocol token；decoder 使用 `Sequence([ByteFallback(), Fuse()])`。
5. 门禁覆盖固定 ID、added-token 原子性、pre-token tiling、`encode → decode` 无损、256 byte vocab 和 save/reload。需要完整 offset 时，只能在固定 PyPI binding 上验证，不能用当前 core 结果代替。
6. 通过后再按部署协议增加自动 BOS/EOS template；不要让模型调研中的 chat-template 加词与 Rust post-processing 混为一谈。

实测口径与复测脚本见 [pretok-performance.md](./pretok-performance.md)。

## 9. 验证入口

- Rust core：在 `tokenizers/` 运行 `make test`；跨 released implementation 的 encode/decode parity 用 `make oracle`；定向测试用 `cargo test -p tk-encode <filter>` 或 `cargo test -p tk-serialize <filter>`。
- Python：在 `bindings/python/` 运行 `make test`；`make check-style` 会重建 stub 并可能修改生成文件，运行后检查 diff。
- Node：在 `bindings/node/` 先执行 `yarn build`，再运行 `make test`。
- 当前重点回归：single/pair ID、decode round-trip、special 输入匹配与模板加词、truncation override、padding mask、legacy→canonical→reader，以及 reader/writer 往返。offset、stride/overflowing 和 trainer 回归属于尚未迁入的边界，不应列为当前 core 已通过能力。
