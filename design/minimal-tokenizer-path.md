# 自研 Tokenizer 最小路径：实施步骤

> 本文件将 `tokenizer-architecture.md` §9「自研 tokenizer 准备：最小路径建议」落为**可执行步骤**。
> 核心约束：**不修改仓库源码**——全程用 PyPI 发布的 `tokenizers` 包，不构建本地 Rust core、不改动任何源文件。脚本放在仓库外运行即可。
>
> 配套设计：`uax29-sentence-pretokenizer.md`（预分词正则与行为）、`pretok-performance.md`（性能基准）。

---

## 0. 目标管线（一图速览）

| 组件 | 取值 | 说明 |
|---|---|---|
| 基础词表 | `BASE_SPECIALS` + `initial_alphabet` + 256 个 byte token | 特殊/工具 token、单字符种子、字节兜底分层管理 |
| 特殊/工具 token | `AddedToken(t, special=True, normalized=False)` 提前注册 | 在 normalizer 之前、按原文切分；不吞前后空白 |
| `normalizer` | `None` | 原文直通，大小写/重音/全角/空白/控制字符全保留 |
| `pre_tokenizer` | `Split(CLASS_REGEX, merged_with_next)` | 字符类切分（无句界） |
| `model` | `BPE(byte_fallback=True, unk_token="<unk>")` + `<0x00>..<0xFF>` | 字节回退必须同时存在于 model vocab |
| `post_processor` | `None` | 先跑通全链，再按需加 bos/eos |
| `decoder` | `Sequence([ByteFallback(), Fuse()])` | 推理期还原 |

---

## 0.1 基础词表规范

> 基础词表是**可重建的种子规范**，不是最终 BPE 词表。最终词表仍由语料和 merges 学习得到。

基础词表分三层，不能混成一个列表：

| 层 | 内容 | 注入位置 | 约束 |
|---|---|---|---|
| `BASE_SPECIALS` | `<unk>`、`<pad>`、序列边界、ChatML、工具、多模态协议 | `BpeTrainer.special_tokens` + `add_special_tokens` | 多字符串；固定顺序；`special=True` |
| `initial_alphabet` | ASCII、空白、CJK 标点/假名、组合字符、语料实际字符 | `BpeTrainer.initial_alphabet` | 每项必须是单个 Unicode 码点 |
| `BYTE_TOKENS` | `<0x00>..<0xFF>` | 训练后写入 `model.vocab` | 不能放入 `initial_alphabet`；不是 `AddedToken` |

`SPECIALS`、字符种子、字节 token 三者职责不同。把 `<|tool_call|>` 放进 `initial_alphabet` 会被当成多字符串并只取首字符；把 `<0x41>` 放进 `initial_alphabet` 也不会得到真正的字节回退。

---

## 1. 环境准备（一次性）

```bash
# 用发布的 tokenizers，而非本地仓库构建 → 天然"不修改仓库代码"
python -m venv ~/min_tok.venv && source ~/min_tok.venv/bin/activate
pip install "tokenizers==0.23.2"   # 固定 byte token 格式与 AddedToken 行为
```

> 仓库的 `design/` 文档描述的是**源码内部行为**（`Split` 的 `sentence_breaks`、Rust 单测等）。本步骤只消费其结论（CLASS_REGEX 字符串 + 管线取值），不依赖仓库源码。若想复现 `sentence_breaks` 句界扩展，才需要触碰 `tokenizers/src/pre_tokenizers/split.rs` 等源码，本最小路径不需要。

---

## 2. 实施步骤（对应 §9 五点）

| §9 条目 | 落地动作 | Python 取值 |
|---|---|---|
| 基础词表 | 固定 `BASE_SPECIALS`、语料字符 alphabet、256 个 byte token | 三层分开注册/注入 |
| 抽取特殊/工具 token | 训练前 `add_special_tokens`；同序传给 `BpeTrainer.special_tokens` | `AddedToken(t, special=True, normalized=False)` |
| 规范化 minimal/identity | 显式置 `None`，原文直通 | `tok.normalizer = None` |
| 预分词 | 字符类 `Split`（无句界） | `Split(Regex(CLASS_REGEX), "merged_with_next")` |
| 分词模型 | BPE + 完整 byte fallback 词表 | `BPE(byte_fallback=True, unk_token="<unk>")` + `<0x00>..<0xFF>` |
| post_processor | 先 `None`，跑通全链再按需加 | `tok.post_processor = None` |

### 2.1 完整训练脚本（保存为 `~/min_tok.py`，**放在仓库外**）

```python
import json
import re
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import ByteFallback, Fuse, Sequence

# ── 1. 字符类预分词正则（来自 uax29-sentence-pretokenizer.md §2，含 JUNK 拖尾）──
CLASS_REGEX = r"""
\p{White_Space}*
(?:
  [\p{Han}\p{Hiragana}\p{Katakana}\u30FC]+
    (?:[^\p{White_Space}\p{L}\p{N}]*[\p{Han}\p{Hiragana}\p{Katakana}\u30FC]+)*
| [\p{Latin}]+
    (?:[^\p{White_Space}\p{L}\p{N}]*[\p{Latin}]+)*
| \p{N}+
    (?:[^\p{White_Space}\p{L}]*\p{N}+)*
)
"""

# ── 2. 基础词表：特殊/工具 token ──
# 顺序即 ID 约定；<unk> 必须第一位。新增 token 时追加，不要插入中间。
CORE_SPECIALS = [
    "<unk>", "<pad>", "<s>", "</s>",
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
]
TOOL_SPECIALS = [
    "<|tools_begin|>", "<|tools_end|>",
    "<|tool_call|>", "<|/tool_call|>",
    "<|tool_result|>", "<|/tool_result|>",
    "<|tool_error|>",
]
MULTIMODAL_SPECIALS = [
    "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>",
    "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
    "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
]
OPTIONAL_SPECIALS = []  # 例如 FIM、<think>、厂商兼容 token
BASE_SPECIALS = [
    *CORE_SPECIALS,
    *TOOL_SPECIALS,
    *MULTIMODAL_SPECIALS,
]
SPECIALS = [*BASE_SPECIALS, *OPTIONAL_SPECIALS]

# 字节 token 属于 model vocab，不属于 AddedToken/initial_alphabet。
# tokenizers 0.23.2 与 InternLM2/Yi 缓存采用 <0xHH> 格式。
BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))

# ── 3. 语料字符种子 ──
CORPUS_PATH = Path("/path/to/your/corpus.txt")
ALPHABET_PATH = Path("/path/to/your/alphabet.json")  # 可选；通常指向 alphabet.json


def iter_corpus(path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def build_initial_alphabet(corpus_path, alphabet_path=None):
    # 固定基础字符：ASCII、Latin-1、组合字符、通用标点、CJK/假名、全角。
    chars = {chr(cp) for cp in range(0x20, 0x7F)}
    chars.update("\n\r\t")
    for start, end in (
        (0x00A0, 0x0100),  # Latin-1 可见字符
        (0x0300, 0x0370),  # 组合音标
        (0x2000, 0x2070),  # 通用标点/格式
        (0x3000, 0x3100),  # CJK 标点、平假名、片假名
        (0xFE00, 0xFE10),  # 变体选择符
        (0xFF01, 0xFF60),  # 全角 ASCII/标点
    ):
        chars.update(chr(cp) for cp in range(start, end))

    # 语料统计优先于手工范围；已有 alphabet.json 时避免重复扫描大文件。
    if alphabet_path is not None and alphabet_path.exists():
        chars.update(json.loads(alphabet_path.read_text(encoding="utf-8")))
    for line in iter_corpus(corpus_path):
        chars.update(line)

    # initial_alphabet 每项必须是单个 Unicode scalar。
    return sorted(
        ch for ch in chars
        if len(ch) == 1
        and not 0xD800 <= ord(ch) <= 0xDFFF
        and not 0xFDD0 <= ord(ch) <= 0xFDEF
        and ord(ch) not in (0xFFFE, 0xFFFF)
    )


base_alphabet = build_initial_alphabet(CORPUS_PATH, ALPHABET_PATH)
initial_alphabet = list(base_alphabet)
assert base_alphabet == sorted(set(base_alphabet))
assert all(len(ch) == 1 for ch in base_alphabet)
assert len(BASE_SPECIALS) == len(set(BASE_SPECIALS))


# ── 4. 组装最小管线 ──
tok = Tokenizer(BPE(byte_fallback=True, unk_token="<unk>"))
tok.normalizer = None
tok.pre_tokenizer = Split(Regex(CLASS_REGEX), behavior="merged_with_next")
tok.post_processor = None
for token in SPECIALS:
    tok.add_special_tokens([
        AddedToken(
            token,
            special=True,
            normalized=False,
            single_word=False,
            lstrip=False,
            rstrip=False,
        )
    ])

# ── 5. 训练器 ──
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=SPECIALS,
    initial_alphabet=initial_alphabet,
    show_progress=True,
)

# train_from_iterator 消费一次；这里传入可重新创建的生成器，alphabet 扫描独立完成。
tok.train_from_iterator(iter_corpus(CORPUS_PATH), trainer)


# ── 6. 注入 256 个 byte fallback token，再设置 decoder ──
def inject_byte_tokens(tokenizer):
    data = json.loads(tokenizer.to_str())
    model = data["model"]
    vocab = model["vocab"]
    next_id = max(vocab.values(), default=-1) + 1
    for token in BYTE_TOKENS:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    model["byte_fallback"] = True
    model["unk_token"] = "<unk>"
    return Tokenizer.from_str(json.dumps(data, ensure_ascii=False))


tok = inject_byte_tokens(tok)
tok.decoder = Sequence([ByteFallback(), Fuse()])

# ── 7. 落盘 + 往返无损断言 ──
tok.save("tokenizer.json")
reloaded = Tokenizer.from_file("tokenizer.json")
serialized = json.loads(reloaded.to_str())
assert serialized["normalizer"] is None, "normalizer 必须是 null"
assert serialized["post_processor"] is None, "post_processor 必须是 null"
assert serialized["model"]["byte_fallback"] is True, "byte_fallback 必须为 True"
assert reloaded.token_to_id("<unk>") == 0
special_ids = [reloaded.token_to_id(token) for token in SPECIALS]
assert len(special_ids) == len(set(special_ids))
assert special_ids == list(range(len(SPECIALS)))  # 自训路径的固定 ID 约定

byte_vocab = {
    token for token in reloaded.get_vocab()
    if re.fullmatch(r"<0x[0-9A-F]{2}>", token)
}
assert byte_vocab == set(BYTE_TOKENS), "必须包含完整 256 个 byte token"

sample = "Hello 你好，世界 123 abc"
enc = reloaded.encode(sample, add_special_tokens=False)
assert reloaded.decode(enc.ids, skip_special_tokens=False) == sample

# 工具协议标记必须原子匹配，且不被前后空白吞掉。
tool_sample = "<|im_start|>user<|im_end|><|tool_call|>get_weather<|/tool_call|>"
tool_enc = reloaded.encode(tool_sample, add_special_tokens=False)
for token in ("<|im_start|>", "<|im_end|>", "<|tool_call|>", "<|/tool_call|>"):
    assert reloaded.token_to_id(token) in tool_enc.ids
assert reloaded.decode(tool_enc.ids, skip_special_tokens=False) == tool_sample

# ── 8. （可选第二步）指令微调自动加尾词 ──
from tokenizers.processors import TemplateProcessing

reloaded.post_processor = TemplateProcessing(
    single="$0 </s>",
    special_tokens=[("</s>", reloaded.token_to_id("</s>"))],
)
```

> 参考 `dataset/pretok_cmp/alphabet.json` 时，优先使用已统计的语料字符。若要补充 `InternLM2`/`Yi` 的字符，只取 `len(token) == 1` 的条目；不要合并 `ByteLevel` 的 `Ġ/Ċ` 或 SentencePiece 的 `▁`。

### 2.2 验证命令

```bash
source ~/min_tok.venv/bin/activate
python ~/min_tok.py                      # 训练、注入 byte token、结构/协议/无损断言
python - <<'PY'                          # 独立复验序列化往返
import json
j = json.load(open("tokenizer.json"))
assert j["normalizer"] is None
assert j["post_processor"] is None
assert j["model"]["byte_fallback"] is True
assert len([t for t in j["model"]["vocab"] if t.startswith("<0x")]) == 256
print(j["normalizer"], j["post_processor"], j["model"]["byte_fallback"])
PY
```

---

## 3. 关键坑位（来自设计文档 §6 / §9）

1. **三层词表不能混用**：`SPECIALS` 是多字符串协议 token；`initial_alphabet` 只能是单 Unicode 码点；`<0xHH>` 是 model vocab 条目。把协议 token 或 byte token 传给 `initial_alphabet` 不会得到预期词表。
2. **特殊 token 必须双重声明**：既要在 `tok.add_special_tokens()` 中注册，也要把同一有序列表传给 `BpeTrainer.special_tokens`。`<unk>` 放第一位；落盘后检查 ID 唯一且未改变。
3. **`byte_fallback` 不会自动生成字节词表**：训练后显式追加 `<0x00>..<0xFF>`，设置 `model.byte_fallback=true`，再 `Tokenizer.from_str/from_file` 重建。当前 PyPI `0.23.2` 使用小写 `x`、大写 hex 数字的 `<0xHH>` 格式。
4. **`unk_token="<unk>"` 兜底**：必须把 `<unk>` 放入 `SPECIALS`。完整 byte 词表存在时，正常 UTF-8 输入不应走 `<unk>`；`<unk>` 只保留给不可表示或异常输入。
5. **工具 token 是协议边界**：工具名称、JSON 参数、结果正文不要加入特殊词表；只保证 begin/end、call/result、error 标记原子化。`lstrip/rstrip` 必须关闭，否则会吞掉协议外空白并破坏 offsets。
6. **`post_processor` 先 `None`**：`ByteLevel(trim_offsets=true)` / `RobertaProcessing` 会改 offsets，破坏 UAX29 无损约定；`TemplateProcessing/BertProcessing` 会自动加 CLS/SEP，与「手动控制」冲突。先跑通 `raw→split→BPE→Encoding→decode` 全链 + offset 单测，再按需加 eos。
7. **`merged_with_next` 的合并方向**：每个 match 与**后置** gap 合并（前置 gap 独立 piece），如 `Hello, world! → ['Hello,', ' world!']`。断言不要用 span 直接比对，改用「拼接==输入 + 去空白后核心 token 顺序」或「offset 连续单调」。
8. **流式训练防 OOM**：整份语料常驻 + trainer 词频表曾触发 OOM；务必 `train_from_iterator` 逐行，字符种子单独扫描或读取 `alphabet.json`，不要把整份语料常驻内存。
9. **缓存词表不能整份并入**：30 多份缓存大多是 ByteLevel；只从 `InternLM2`/`Yi` 等字符级 BPE 参考词表取单字符，并过滤 `▁`、`Ġ`、`Ċ` 等实现专用 token。

---

## 4. 后续（按需）

- 需要句界隔离时，开启 `sentence_breaks=True`（仅仓库源码支持，见 `uax29-sentence-pretokenizer.md` §10）。
- 性能基准参考 `pretok-performance.md`；编码成本主要在预分词阶段（C2 约 11/6.5/4.7 MiB/s，release 构建）。
- 升级 `tokenizers` / 换正则引擎时，重跑 §2.2 的序列化与无损断言门禁。

---

## 5. 复用现有字符级 BPE 词表（免重训）

> 景观调查覆盖 23 个家族；当前 HF 缓存中可解析约 32 份 `tokenizer.json`。绝大多数是 byte-level 实现——`pre_tokenizer` 尾部通常为 `ByteLevel`，词表是 byte→unicode 映射（`Ġ`/`Ċ` 等），与本方案的字符级偏好不符。
> 但 **character-level + BPE 存在**，即 **SentencePiece-BPE 系**：词表以 Unicode 字符为字母表、`byte_fallback=True` 兜底，结构与本方案 `BPE(byte_fallback=True, unk_token="<unk>")` **字面同构**。可直接拿其词表复用，**免重训**。

### 5.1 三个候选（已软链到 `dataset/pretok_cmp/landsurvey/`，与 byte-level 覆盖集并列）

| 家族 | 仓库 | model | byte_fallback | 词表 | 单字汉字 | 主文件 |
|---|---|---|---|---|---|---|
| `baichuan2` | `baichuan-inc/Baichuan2-7B-Base` | **SPM BPE** | True | 125,696 | **20,338 (16.2%)** | `tokenizer.model` |
| `internlm2` | `internlm/internlm2-1_8b` | BPE | True | 92,544 | 7,047 (7.6%) | `tokenizer.json` |
| `yi` | `01-ai/Yi-1.5-6B` | BPE | True | 63,992 | 4,019 (6.3%) | `tokenizer.json` |

三者 `pre_tokenizer` 均为 `null`（字符级，无 `ByteLevel`）；词表都含 256 个 `<0xXX>` 字节兜底 token（`baichuan2` 已确认含 `<0x00>`）。本地复验：

```text
internlm2 : model.type=BPE byte_fallback=True pre_tokenizer=null vocab=92544  单字汉字=7047 (7.6%)
yi        : model.type=BPE byte_fallback=True pre_tokenizer=null vocab=63992  单字汉字=4019 (6.3%)
baichuan2 : SPM model_type=BPE vocab=125696 byte_fallback=True 单字汉字=20338 (16.2%)  含<0x00>
```

### 5.1.1 缓存审计边界

当前 HF 缓存可扫描约 32 份 `tokenizer.json`，`landsurvey/` 保留 13 个稳定软链。用法分三类：

- **特殊/工具 token 审计**：统计 `added_tokens` 中 `special=true` 的协议名；不复制 `<SPECIAL_N>`、reserved、dummy、placeholder 等占位 token。
- **字符级 BPE 参考**：只从 `internlm2`、`yi` 的 model vocab 提取 `len(token) == 1` 的条目；`baichuan2` 作为 CJK 覆盖参考，但需处理 SPM `▁`。
- **byte-level 词表**：只用于行为/覆盖率对照，不把 `Ġ`、`Ċ` 等映射字符并入 `initial_alphabet`。

缓存用于选择命名和验证覆盖率，不用于无条件重排已有 ID。复用时保留原 model vocab/merges，仅追加缺失特殊 token，并检查冲突。

代表协议对照：

| 缓存家族 | 代表 token | 本方案处理 |
|---|---|---|
| Qwen / MiMo | ChatML + `<|object_ref_*|>`、`<|box_*|>`、`<|vision_*|>` | 纳入默认 `MULTIMODAL_SPECIALS` |
| Mistral / Nemotron | `[AVAILABLE_TOOLS]`、`[TOOL_RESULTS]`、`[TOOL_CALLS]` | 只参考边界语义，转换为本方案 `TOOL_SPECIALS` |
| Inkling | `<|content_invoke_tool_json|>`、`<|content_tool_error|>` | 作为厂商 alias，不直接混入默认 ID |
| MiniMax | `<function_call>`、`<code_interpreter>` | 作为领域扩展，不当作通用协议 |

缓存中的不可见 U+200B 等兼容字符不作为默认命名；默认工具 token 使用可见 ASCII 标记，避免协议字符串难以排查。

### 5.2 选型

- **中文优先（CJK 导向，契合 CLASS_REGEX）** → `baichuan2`：20,338 个单字汉字、含汉字 piece 占 56%，中文无损率最好、词表最大。代价：只有 SPM `tokenizer.model`，需 `sentencepiece`（已装）或 `transformers` 加载，并一次性转成 `tokenizer.json`。
- **免转换 + 宽松许可** → `yi`（Apache-2.0，63,992）或 `internlm2`（92,544），二者均带 `tokenizer.json`，`Tokenizer.from_file` 直接可用。
- 三者构件均满足 `model = BPE(byte_fallback=True)`，与 §0/§2 的 `model` 取值**字面一致**；复用时保留其已有 256 个 byte token，再追加本方案缺失的特殊/工具 token，不重排原 ID。

### 5.3 复用步骤（免重训，只换管线）

1. 载入词表（`baichuan2` 走 SPM；`internlm2`/`yi` 走 `Tokenizer.from_file("tokenizer.json")`）。
2. 保留原 model 的 `byte_fallback`、256 个 byte token、merges 和已有特殊 token；只追加缺失的 `BASE_SPECIALS`。
3. `tok.normalizer = None`。SPM 系把空格烧成 `▁`，须一并去掉（见 §5.4 坑位④）。
4. `tok.pre_tokenizer = Split(Regex(CLASS_REGEX), behavior="merged_with_next")` 替换原 `null`/`Metaspace`。
5. **`▁` 重写**：SPM 词表 piece 带 `▁` 前缀（如 `▁你好`），而 `normalizer=None` 下输入是真实空格，匹配不上 `▁` piece。必须把 key 做 **`▁X → " X"`** 重写（或保留 `Replace(" "→"▁")`，但那就偏离 `normalizer=None`）。
6. 对新增特殊 token 使用 `AddedToken(special=True, normalized=False, single_word=False, lstrip=False, rstrip=False)`，不重排已有 ID。
7. 保留 `byte_fallback=True`、`unk_token="<unk>"`，`decoder = Sequence([ByteFallback(), Fuse()])`。
8. 落盘后跑 §2.1 的结构、特殊 token、协议和无损断言门禁。

### 5.4 关键坑位（承接 §3）

1. **`▁` 与 `normalizer=None` 冲突**：SPM 系字符表以 `▁` 标记词边界，与本方案 `normalizer=None`（空格原样）不兼容。拿到词表后必须 `▁X → " X"` 重写 piece key，否则中/英文 token 全部失配。

### 5.5 下载

```bash
cd dataset/pretok_cmp && \
HF_ENDPOINT=https://hf-mirror.com \
  /home/zmz/deepsleep/tokenizers/tmp/.venv/bin/python -P download_charbpe_3.py
```

产物：`landsurvey/{baichuan2,internlm2,yi}` 软链 → HF 缓存快照（真实文件在 `~/.cache/huggingface/hub`，无实体落盘）。

---

## 6. 在初始词表上扩展训练（方案 A：种子注入 + 续训）

> 适用：已有 `tokenizer.json`（§2 自训产物，或 §5 复用的 char-BPE 词表），想用**新语料**扩充并重新学合并规则——既非从零训（§2），也非仅换管线免重训（§5）。

### 6.1 约束（务必先读）

- **HF `tokenizers` 的 BPE 不支持真正的增量续训**：`BpeTrainer.train_from_iterator()` 每次都从语料**重建 vocab + merges**，不会在旧合并规则上「接着学」。因此「在初始词表上扩展训练」只能落地为：**把初始词表降解成字符种子注入，再在新语料上重训**。
- `initial_alphabet` 接受的是**单字符列表**，不是多字符 token。库的 alphabet 本质是一组基础字符，BPE 只在字符之上学合并；传多字符串无效（§3 坑位①已述：`initial_alphabet` 只取每项首字符）。

### 6.2 定义初始词表（字符级）

从已有 `tokenizer.json` 的 vocab 抽字符，并并入新语料字符，保证覆盖：

```python
from tokenizers import Tokenizer

# SPECIALS / base_alphabet 沿用 §2.1 的定义。
old = Tokenizer.from_file("dataset/pretok_cmp/tokenizer_C2_nosb_sameclass.json")
chars = set(base_alphabet)
for token in old.get_vocab():
    if token in SPECIALS or token.startswith("<0x"):
        continue
    chars.update(token)                # 只拆普通 vocab 条目中的字符
for line in new_corpus():
    chars.update(line)                 # 并入新语料字符
chars.discard("▁")                     # identity 模式不引入 SPM 词边界
initial_alphabet = sorted(ch for ch in chars if len(ch) == 1)
```

> 实仓写法见 `dataset/pretok_cmp/train_and_compress.py` 的 `cmd_train`：`alpha = set(); for line in iter_lines(): alpha.update(line)`，再 `initial_alphabet=sorted(alpha)`。扩展训练时额外排除特殊 token、byte token 和 `▁`，避免把协议/实现字符混入基础 alphabet。

### 6.3 使用：种子注入 + 续训

```python
from tokenizers import Tokenizer, Regex, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import ByteFallback, Fuse, Sequence

# SPECIALS、BASE_SPECIALS、CLASS_REGEX、initial_alphabet 沿用 §2.1。
tok = Tokenizer(BPE(byte_fallback=True, unk_token="<unk>"))
tok.normalizer = None
tok.pre_tokenizer = Split(Regex(CLASS_REGEX), behavior="merged_with_next")
tok.post_processor = None
for token in SPECIALS:
    tok.add_special_tokens([
        AddedToken(
            token,
            special=True,
            normalized=False,
            single_word=False,
            lstrip=False,
            rstrip=False,
        )
    ])

trainer = BpeTrainer(
    vocab_size=32000,                  # 比旧词表略大，给新 token 留空间
    min_frequency=2,
    initial_alphabet=initial_alphabet, # ← §6.2 定义的字符种子
    special_tokens=SPECIALS,            # 不再只传 <unk>
    show_progress=True,
)
# 在「新语料」上训练；若想保留旧 token 词频，把旧语料也并入迭代器
tok.train_from_iterator(new_corpus(), trainer)
tok = inject_byte_tokens(tok)          # 复用 §2.1 的注入函数
tok.decoder = Sequence([ByteFallback(), Fuse()])
tok.save("tokenizer_extended.json")
```

### 6.4 坑位（承接 §3）

1. **旧多字符 token 不自动保留**：`initial_alphabet` 只保「字符」，不保「词」。旧 token（如 `你好`）只有当新语料里仍以足够频率出现时才会被重新学出。要兼顾旧词频，把**旧语料也喂进** `train_from_iterator` 的迭代器（与 `train_and_compress.py` 的 `iter_lines()` 同口径）。
2. **特殊/工具 token 不从旧 vocab 推导 alphabet**：从 §2.1 的 `SPECIALS` 统一注册；不要把旧特殊 token 的内部字符当作新的基础字符。
3. **字节回退 token 无法用 alphabet 注入**：训练后仍需做 `tokenizer.json` 手术，注入 `<0x00>..<0xFF>` 并置 `model.byte_fallback=true`，见 §3 坑位③。
4. **vocab_size 预留余量**：扩展训练词表通常比旧词表大，设 `vocab_size` 时给新 token 留空间，否则新语料的高频子词会被截断。
5. **非破坏性替代见 §6.5**：若只想新增少量领域词/专名、保旧词表稳定，改用 `add_tokens()`；特殊/工具 token 不走该路径。

### 6.5 对照：仅追加 token（方案 B，非破坏性）

```python
tok = Tokenizer.from_file("dataset/pretok_cmp/tokenizer_C2_nosb_sameclass.json")
added = tok.add_tokens(["新词1", "专用术语XYZ"])   # 返回实际新增条数
print("新增:", added, "|V|:", tok.get_vocab_size())
tok.save("tokenizer_extended.json")
```

代价：新 token 为整块单 token，不参与子词合并，压缩率略吃亏，但覆盖与语义确定；旧词表原样保留。`add_tokens()` 只用于普通领域词/专名；特殊/工具 token 必须使用 `add_special_tokens()`，否则会被当作普通文本参与 BPE。

### 6.6 选型小结

| 目标 | 方案 | 出处 |
|---|---|---|
| 新语料很大，重学最优合并、扩充词表 | **A**（种子注入 + 续训，旧语料并入） | §6 |
| 仅加少量领域词/专名，旧词表稳定 | **B**（`add_tokens`） | §6.5 |
| 直接复用现成词表、只换管线 | 免重训复用 | §5 |
