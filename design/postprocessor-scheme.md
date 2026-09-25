# Tokenizer 后处理方案：当前契约与主流模型对照

## 1. 结论

- 当前 encode runtime 只有 `PipelinePostProcessor { single, pair }`；两份配置都是 `Template`，布局固定为 `prefix / A / infix / B / suffix`。
- `post_processor: null`（或缺省）读成 `PipelinePostProcessor::default()`：不添加特殊 ID；pair 仍可保留 A=`0`、B=`1` 的 `type_ids`。
- 当前 `tk-serialize` 只读 canonical `TemplateProcessing`。`ByteLevel`、`BertProcessing`、`RobertaProcessing`、`Sequence` 以及旧式 `special_tokens` 名称表，都是 `tk-convert` 负责降级的历史形态。
- 顶层 `added_tokens` 仍是词表/输入匹配配置，不等于旧 `PostProcessor::added_tokens()` 方法；两者不要混称。
- decoder-only 模型的 chat BOS/EOS 通常由 `chat_template`、`tokenizer_config.json` 或上层 `transformers.update_post_processor()` 决定。不能只看 Rust post-processor 就推断最终 chat 输出。

当前管线：

```text
Input -> Normalizer -> PreTokenizer -> Model -> PipelinePostProcessor -> Encoding
```

当前 `PipelineTokenizer` 的 `Encoding` 主要是 `ids`、`type_ids` 和 `attention_mask`；旧 `process_encodings()` 写入的 offset、`special_tokens_mask`、`sequence_id` 语义不属于当前 runtime 契约。

## 2. 五种历史配置形态

| 历史 `post_processor` 形态 | 旧配置要点 | `tk-convert` 后的当前形态 |
|---|---|---|
| `TemplateProcessing` | `single`/`pair` 由 `Sequence`、`SpecialToken` 组成，特殊词名在 `special_tokens` 表中查找 | 展平为 `{"seq":"A"|"B"}`、`{"ids":[...]}`，删除名称表；`tk-serialize` 只接受这种 canonical 写法 |
| `BertProcessing` | `cls`、`sep` 是 `[token, id]` 对 | `TemplateProcessing`：single=`[CLS] A [SEP]`；pair=`[CLS] A [SEP] B:1 [SEP]:1`，加词数 2/3 |
| `RobertaProcessing` | `cls`、`sep` 对；旧实现还带 `add_prefix_space`、`trim_offsets` | `TemplateProcessing`：single=`<s> A </s>`；pair=`<s> A </s></s> B </s>`，全部 type id 为 0，加词数 2/4；旧 offset 修正不保留 |
| `ByteLevel` | `add_prefix_space`、`trim_offsets`、可选 `use_regex` | 裸成员降为 `null`/默认 frame，不添加 token；在 `Sequence` 中是无操作成员。旧实现的 `set_sequence_id` 与 offset retagging 不进入当前 pipeline |
| `Sequence` | `processors` 数组，可混合上述成员 | 递归降级后保留唯一真正加词的成员；没有加词成员时保留第一个已降级成员（canonical writer 对默认 frame 写 `null`），空数组或多个加词成员报错 |

这五种是历史配置形态，不是五种当前 `PipelinePostProcessor` 变体。canonical reader 明确拒绝旧名称；先经过 `tk-convert`，再交给 `tk-serialize`。

## 3. 当前 `Template` 契约

| `Template` 字段 | 当前含义 |
|---|---|
| `prefix` | A 之前插入的特殊 ID |
| `infix` | A 与 B 之间插入的特殊 ID；single 始终为空 |
| `suffix` | 最后一条序列之后插入的特殊 ID |
| `a_type_id` | A 的 type id，默认 0 |
| `b_type_id` | B 的 type id；只有 pair 模板设置，缺省 pair 为 1 |

执行顺序在 `tokenizers/tk-encode/src/tokenizer/pipeline/post_processor.rs`：

1. 根据输入形状选择 `single` 或 `pair`。
2. `n_special()` 计算 `prefix + infix + suffix` 的 ID 数量。
3. 只有 `EncodeOptions::add_special_tokens` 为 `true` 时，才把这批特殊 ID 纳入截断预算并插入。
4. `post_process::<SPECIALS>` 写入 ID；`SPECIALS` 为 false 时只保留序列本身。`type_ids` 仅在存在非零序列/特殊词标记时分配。

`single` 模板不能引用 B；`pair` 必须按 A、B 顺序引用。历史 trait 的 `added_tokens(is_pair)` 只负责返回 special 数量，`process_encodings(...)` 只负责把 special ID、offset、mask 等写入旧 `Encoding`；当前实现没有这两个调用点，分别由 `n_special()` 与 `Template::post_process()` 取代。

## 4. 主流模型对照

下表中的 `ByteLevel` 等是 Hugging Face 旧版 `tokenizer.json` 的历史 `type`，不是当前 canonical reader 直接接受的类型。`tk-convert` 后再由 `tk-serialize` 构建 runtime。

| 模型 | HF 历史 `post_processor.type` / 模板 | 当前 Rust pipeline 结果 |
|---|---|---|
| Qwen2-7B / Qwen2.5-7B / Qwen3-0.6B | `ByteLevel`：`add_prefix_space:false, trim_offsets:false, use_regex:false` | `null`/默认 frame；加词数 0；chat 层按需加 EOS/对话标记 |
| DeepSeek-V3 / V2-Lite | `ByteLevel`：`add_prefix_space:true, trim_offsets:false, use_regex:true` | `null`/默认 frame；加词数 0；`use_regex` 不由当前 post-processor 执行 |
| GPT-2 | `ByteLevel`：`add_prefix_space:true, trim_offsets:false`（`use_regex` 缺省为 true） | `null`/默认 frame；加词数 0 |
| Mistral-7B-v0.1 | `TemplateProcessing`：`single=[<s>, A]`，`pair=[<s>, A, <s>:1, B:1]` | 保留模板；`add_special_tokens=true` 时 single 加 1 个 BOS，pair 加 2 个 |
| `bert-base-uncased` / `albert-base-v1` | `TemplateProcessing` / BERT frame：`[CLS] + A + [SEP]` | 保留模板；single 加 2 个，pair 加 3 个 |
| `roberta-base` | `RobertaProcessing`：`cls:<s>, sep:</s>` | 保留转换后的模板；single 加 2 个，pair 加 4 个；旧 offset 修正不执行 |
| `t5-small` | `TemplateProcessing`：`single=A </s>`，`pair=A </s> B </s>` | 保留 EOS 模板；single 加 1 个，pair 加 2 个 |

其他 SentencePiece/模板变体的 EOS 形态依具体 `tokenizer.json` 而定；应解析 `single`/`pair`，不要从模型名推断，也不要把 Metaspace 预分词和 post-processor 混为一谈。

## 5. 关键参数边界

- `add_special_tokens`：`EncodeOptions` 当前参数，默认 `true`；为 false 时不插入 `prefix`/`infix`/`suffix`，截断预算中的特殊词数也为 0。
- `single`、`pair`、`seq`、`ids`、`type_id`：当前 canonical 模板字段；只有模板里的 `ids` 真正增加特殊 token。
- `special_tokens`、`SpecialToken`、`Sequence` 名称、`cls`/`sep`：历史配置语法。`tk-convert` 将名称解析成 ID 后删除这些旧字段。
- `add_prefix_space`、`trim_offsets`、`use_regex`：历史 `ByteLevel` post-processor 字段；整个对象当前降为默认 frame，因此不产生当前 post-processing 行为。`use_regex` 的预分词含义应到 `tk-convert` 的 ByteLevel lowering 中解释。
- 顶层 `added_tokens`：控制特殊/added token 的词表注册与输入切段；它在 Model 前被提取，不是模板 special ID 的同义词。

## 6. 验证路径与入口

实现与测试路径：

- 当前模板与截断预算：`tokenizers/tk-encode/src/tokenizer/pipeline/post_processor.rs`、`tokenizers/tk-encode/src/tokenizer/pipeline/mod.rs`。
- canonical reader/writer：`tokenizers/tk-serialize/src/from_json/post_processors.rs`、`tokenizers/tk-serialize/src/to_json/post_processors.rs`。
- 五种历史形态降级：`tokenizers/tk-convert/src/convert.rs`；覆盖用例在 `tokenizers/tk-convert/tests/convert.rs`。
- reader/writer 回归：`tokenizers/tk-serialize/src/from_json/tests.rs`、`tokenizers/tk-serialize/src/to_json/tests.rs`。

从 `tokenizers/` 执行：

```bash
cargo test -p tk-encode
cargo test -p tk-serialize --features serialize
cargo test -p tk-convert
```

验证单个模型时，先用 `tk_convert::canonicalize_str` 或 `canonicalize_file` 处理 HF `tokenizer.json`（只验证一个节点时可用 `canonicalize_post_processor`），再用 `tk_serialize::from_json` 读取。对同一文本比较 single/pair、`EncodeOptions::default()` 与 `EncodeOptions::no_specials()` 的 IDs、`type_ids`、截断长度和 decode 结果；另测模板关闭时模板 special ID 数为 0（输入中的顶层 added token 另行验证）。模型配置还需同时检查 `chat_template`、`tokenizer_config.json` 的 `eos_token`/`add_eos_token`，但不要把 chat 层加词当作 Rust post-processor 行为。
