# 自研 Tokenizer 最小路径

> 目标：给出不改仓库源码、可直接运行的自训与复用路径。使用 PyPI `tokenizers==0.23.2`，脚本放在仓库外；预分词见 `uax29-sentence-pretokenizer.md`，性能见 `pretok-performance.md`。

## 1. 最终管线与基础词表规范

基础词表分四层，职责不可混用；BPE 最终词表仍由语料和 merges 学习：

| 层 | 内容 | 注入方式 | 约束 |
|---|---|---|---|
| `SPECIALS` | 通用边界与预留槽位、Qwen ChatML/视觉、DeepSeek BOS/EOS/视觉 token | `add_special_tokens()` + `BpeTrainer.special_tokens` | `special=True, normalized=False`；`<unk>` 固定为 ID 0 |
| `PROTOCOL_TOKENS` | 通用 think/tool_call/tool_response 与预留槽位，加 Qwen FIM/repo、DeepSeek 角色/DSML/FIM/repo | 训练后 `add_tokens()` | `special=False, normalized=False`；仍按原文原子匹配，但不会被 `skip_special_tokens=True` 删除 |
| `initial_alphabet` | 基础字符、语料字符、已有词表拆出的单字符 | `BpeTrainer.initial_alphabet` | 每项必须是单个 Unicode scalar |
| `BYTE_TOKENS` | `<0x00>..<0xFF>` | 训练后写入 `model.vocab` | 不是 `AddedToken` 或 alphabet 项 |

`AddedToken` 的 `special` 标志不是“是否原子匹配”的开关。Qwen3.8 与 DeepSeek-V4.1 的协议 token 实际位于 `added_tokens` 且 `special=false`；本方案保持此语义：协议 token 原子化，但解码时保留协议文本。为统一 identity 管线，本方案把协议 token 都设为 `normalized=False`；这不保证复刻 DeepSeek 官方 `normalized=True` 行为或 ID。

协议 profile：默认`both`。`GENERIC_SPECIALS` / `GENERIC_PROTOCOL_TOKENS` 在三个 profile 下都注入，含 `<create>`、`</create>`、`<|extra_0..4|>`、`<|extra_5..9|>` 等预留槽位；厂商集合（`QWEN_SPECIALS` / `DEEPSEEK_SPECIALS` 及其协议 token）按 profile 追加。`both` 只是两套集合的并集，不代表兼容任一厂商的原始 ID；要求逐 ID 对齐时直接加载官方 `tokenizer.json`。

其余管线固定为：

- `normalizer=None`：大小写、重音、全角、空白、control 字符原样保留。
- `pre_tokenizer=Split(Regex(CLASS_REGEX), "merged_with_next")`：字符类切分，无句界。
- `model=BPE(byte_fallback=True, unk_token="<unk>")`，且 vocab 含完整 256 byte token。
- `post_processor=None`；门禁通过后再按需加 bos/eos。`decoder=Sequence([ByteFallback(), Fuse()])`。

## 2. 环境

```bash
python -m venv ~/min_tok.venv
source ~/min_tok.venv/bin/activate
pip install "tokenizers==0.23.2"
```

`tokenizers==0.23.2` 固定 byte token 格式与 `AddedToken` 行为。本路径不依赖本地 Rust core。`dataset/pretok_cmp/train_and_compress.py` 与 C1 JSON 中的 `sentence_breaks` 仅是历史产物：当前 checkout 的 `Split` 只读取/写出 `pattern`、`behavior`、`invert`，旧驱动/字段不能恢复句界；最终配置不传该字段。

## 3. 自训脚本

脚本本体：[`min_tok.py`](../min_tok.py)。它流式训练、注入完整 byte vocab、保存 tokenizer，并执行结构、ID、协议和无损门禁。

运行前确认下列常量：

| 常量 | 作用 |
|---|---|
| `TOKEN_PROFILE` | `qwen` / `deepseek` / `both`（脚本当前为 `both`），只影响 `SPECIALS` / `PROTOCOL_TOKENS` |
| `CORPUS_PATH` | 训练语料，逐行流式读取 |
| `ALPHABET_PATH` | 可选的 `alphabet.json`（字符种子），仓库样本见 `dataset/pretok_cmp/alphabet.json` |

```bash
cd ~                     # 脚本可放仓库外；仓库内则 cd 到仓库根
python min_tok.py        # 产物 tokenizer.json 落在 cwd
```

### 3a. 65536 词表与采样训练（16G 内存 / U 盘语料）

`--vocab-size` 指最终 `get_vocab()` 总量（默认 131072），内部反推
`trainer 预算 = target − len(PROTOCOL_TOKENS) − 256`（`BpeTrainer.vocab_size`
是含 specials 的总量上限，protocol 住 `added_tokens` 不进 `model.vocab`，
round-trip 后 ID 自动重排）。总量不符时门禁 loud-fail（碰撞或语料太小未触顶）。

`--corpus` 可指单文件或目录（目录按 `--glob` 递归，默认 `*.txt`；
`.jsonl/.json` 抽 `--text-field` 字段，`.gz` 直读，`.parquet` 用 pyarrow 按 batch
流读 `text` 列）。`--corpus` 可多次指定多源（如英文/中文分目录），按 `--weights`
（如 `96,81`）切分字节预算后顺序流——BPE 只计词频，源间顺序无关；
字母表扫描同样按权覆盖各源，避免单语字母表漏字符。
`.parquet` 可加 `--min-score`（配 `--score-field`，默认 `score`）过滤低分行；
字段缺失时 loud-error 并列出可用列名。确定性排序单遍流，
`--train-gb`（默认 32）、`--alphabet-gb`（默认 2）封顶字节数，
字母表扫描与训练流共享同一文件顺序；stderr 每 512MiB 打进度（U 盘 stall 可见）。
`--min-frequency` 默认 5（GB 级不用 2，省内存并过滤噪声）；
`--shuffle-files --seed` 可打乱文件顺序。产物旁落同名 `.manifest.json`
（含文件清单、流统计、词表组成），复现训练时连同保存。

脚本末尾的门禁依次检查：当前 `Split` 只含 `pattern`/`behavior`/`invert`、`<unk>` 为 ID 0 且 special ID 连续、special/protocol 的 `special`/`normalized` 标志、256 个 `<0xHH>` 齐全、`Split` 切分样例、协议 token 原子匹配且 `skip_special_tokens=True` 仍保留、offset 连续与 decode 无损。

## 4. Chat Template 层

完整设计、固定版本官方来源、集成代码、输入输出契约、响应解析与回归矩阵见 [`chat-templates.md`](chat-templates.md)。本节只固定 tokenizer 与 chat renderer 的边界：

- `TOKEN_PROFILE` 只控制 `SPECIALS` / `PROTOCOL_TOKENS` 词表；`CHAT_PROFILE` 控制 renderer/parser，取值只能是 `qwen3_8` 或 `deepseek_v4_1`。
- `TOKEN_PROFILE="both"` 只是词表并集，不代表存在兼容两家的混合 chat 协议；调用时仍须显式选择一个 `CHAT_PROFILE`。
- Qwen3.8 使用固定 commit 的官方 `chat_template.jinja`，另需 `jinja2>=3.1,<4`；DeepSeek-V4.1 没有 Jinja，必须使用官方 `encoding/encoding.py`，不能凭记忆重写协议。
- renderer 已显式插入协议 special token，编码时必须关闭 tokenizer 的自动加词：

```python
rendered = renderer.render(messages, ...)
encoding = tokenizer.encode(
    rendered.prompt,
    add_special_tokens=False,
)
```

- 多模态 renderer 只生成占位符和有序 media records；图片/视频像素处理仍由 `AutoProcessor` 或模型专用 processor 完成。
- DeepSeek 默认 profile 严格对应 V4.1；V4 Pro 的 `<｜image｜>` 不进入 V4.1 词表。需要 V4 Pro 时另建 `deepseek_v4_pro` profile。

## 5. 关键限制

- 工具名称、JSON 参数、结果正文仍是普通文本；只固定协议边界。Qwen 的 `<tools>`/`<function=...>` 和 DeepSeek 的角色/DSML 扩展按缓存内容选择性加入，不要把全部业务 schema 固化为 special。
- `merged_with_next` 把 match 与后置 gap 合并，例如 `Hello, world! -> ['Hello,', ' world!']`；训练用 `train_from_iterator` 逐行读取，字符种子可从 `alphabet.json` 读取，避免整份语料常驻。
- `ByteLevel(trim_offsets=true)` / `RobertaProcessing` 可能改 offsets；`Bert/Template` 会加 CLS/SEP/BOS/EOS。原型先保持 `post_processor=None`。
- DeepSeek `tokenizer_config.json` 的 `unk_token=null`、pad 字段可能与 `tokenizer.json` 不一致；复用以 `tokenizer.json` 的 `added_tokens`/model vocab 为准。本自训路径仍保留 `<unk>` 作为安全兜底。

## 6. 复用字符级 BPE（免重训）

历史缓存审计约 32 份 `tokenizer.json`，多数是 byte-level。可复用的是 character-level + `byte_fallback`：

| 候选 | 文件 | vocab | 单字汉字 | 结论 |
|---|---|---:|---:|---|
| `baichuan2` | `tokenizer.model` | 125,696 | 20,338（16.2%） | CJK 最强；需 SPM 转 JSON |
| `internlm2` | `tokenizer.json` | 92,544 | 7,047（7.6%） | JSON 可直接加载 |
| `yi` | `tokenizer.json` | 63,992 | 4,019（6.3%） | Apache-2.0；JSON 可直接加载 |

三者均为 BPE + `byte_fallback`，含 256 个 `<0xHH>`。但 `internlm2`/`yi` 虽是 `pre_tokenizer=null`，原 normalizer 会把空格替换成 `▁`；`baichuan2` 也使用 `▁`。因此三者都须在 identity 管线中重写 `▁`。中文优先选 `baichuan2`（汉字 piece 占 56%）；免转换选 `yi` 或 `internlm2`。

### 缓存审计边界

当前 `landsurvey/` 中景观覆盖集 10 个，加本节 3 个候选，共 13 个稳定软链。缓存只用于命名、协议和覆盖率审计：

- Qwen3.8：`SPECIALS` 采用 ChatML、视觉、音频/TTS；tool、FIM、repo、think 走 `PROTOCOL_TOKENS`，且官方 `tokenizer.json` 标记为 `special=false`。
- DeepSeek-V4.1 严格 profile：`SPECIALS` 采用 BOS/EOS/PAD、`end_of_query`、RL 图像/几何 token；DSML、角色、reminder、任务、图像占位符、FIM/repo 兼容项走 `PROTOCOL_TOKENS`，同样为 `special=false`。V4 Pro token 不混入。
- DeepSeek 的大量 `｜place▁holder▁no▁N｜`、reserved/dummy token 不进入基础词表；厂商小版本特有 token 仅在对应缓存中确认后加入。
- 字符参考只取字符级 BPE vocab 中 `len(token) == 1` 的项；三个候选都需处理 `▁`。
- byte-level 的 `Ġ/Ċ` 等映射字符只作行为对照，不并入 `initial_alphabet`。
- 工具标记必须从官方 `added_tokens[].content` 逐字复制。Qwen3.8 的 tool-call opening/closing content 确实含 U+200B；不要手工改成可见 ASCII，也不要凭肉眼重建。

复用步骤：

1. 载入原 tokenizer；保留其 model、merges、256 byte token、已有 specials，不重排 ID。
2. 缺失的 `SPECIALS` 用 `add_special_tokens()`；缺失的 `PROTOCOL_TOKENS` 用 `add_tokens()`，并保持 `special=False`。
3. 设置 `normalizer=None` 与本方案 `Split`；`baichuan2`/`internlm2`/`yi` 的 `▁` 必须同步重写 vocab key 和 merges 两端，并在替换前检查 key 冲突，否则真实空格无法命中。
4. 保留 `byte_fallback=True`、`<unk>` 和 `Sequence([ByteFallback(), Fuse()])`。
5. 检查结构、special/protocol 标志、256 byte token、协议原子性与无损；自训专用的“special ID 连续且 `<unk>`=0”断言不适用于保留原 ID 的复用路径。

下载：

```bash
cd dataset/pretok_cmp
HF_ENDPOINT=https://hf-mirror.com python -P download_charbpe_3.py
```

产物：`landsurvey/{baichuan2,internlm2,yi}` 指向 HF 缓存快照，无实体落盘。

## 7. 扩展训练决策

HF BPE 不支持真正增量续训；`train_from_iterator` 每次重建 vocab 与 merges。扩展训练只能“旧词表拆字符 + 新语料 + 重新训练”：

1. 以 `initial_alphabet` 为底，从旧 model vocab 逐项加入字符，排除 `SPECIALS`、`PROTOCOL_TOKENS`、旧 `added_tokens`、`<0x...>`；再加入新语料字符，删除 `▁`，最终只保留单个 Unicode scalar。
2. 复用 §3 全部管线与 `BpeTrainer(vocab_size=32000, special_tokens=SPECIALS, initial_alphabet=extended_alphabet)`；旧语料也并入 iterator，以保留旧多字符 token 的词频。
3. 训练后用 `add_tokens()` 注册 `PROTOCOL_TOKENS`，再调用 `inject_byte_tokens()`，设置同一 decoder，保存并重跑 §3 门禁。`vocab_size` 需给新语料留余量。

少量领域词/专名且必须保持旧 ID 时，改用 `tok.add_tokens(["新词1", "专用术语XYZ"])`；不重算 merges、压缩率略差。特殊 token 用 `add_special_tokens()`；Qwen/DeepSeek 工具/角色/FIM token 用 `add_tokens()` 并保持 `special=False`。
