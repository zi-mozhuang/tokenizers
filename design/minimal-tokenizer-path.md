# 自研 Tokenizer 最小路径

> 目标：给出不改仓库源码、可直接运行的自训与复用路径。使用 PyPI `tokenizers==0.23.2`，脚本放在仓库外；预分词见 `uax29-sentence-pretokenizer.md`，性能见 `pretok-performance.md`。

## 1. 最终管线与基础词表规范

基础词表分四层，职责不可混用；BPE 最终词表仍由语料和 merges 学习：

| 层 | 内容 | 注入方式 | 约束 |
|---|---|---|---|
| `SPECIALS` | 通用边界、Qwen ChatML/视觉、DeepSeek BOS/EOS/视觉 token | `add_special_tokens()` + `BpeTrainer.special_tokens` | `special=True, normalized=False`；`<unk>` 固定为 ID 0 |
| `PROTOCOL_TOKENS` | Qwen/DeepSeek 工具调用、工具结果、角色、FIM、思考标记 | 训练后 `add_tokens()` | `special=False, normalized=False`；仍按原文原子匹配，但不会被 `skip_special_tokens=True` 删除 |
| `initial_alphabet` | 基础字符、语料字符、已有词表拆出的单字符 | `BpeTrainer.initial_alphabet` | 每项必须是单个 Unicode scalar |
| `BYTE_TOKENS` | `<0x00>..<0xFF>` | 训练后写入 `model.vocab` | 不是 `AddedToken` 或 alphabet 项 |

`AddedToken` 的 `special` 标志不是“是否原子匹配”的开关。Qwen3.8 与 DeepSeek-V4.1 的协议 token 实际位于 `added_tokens` 且 `special=false`；本方案保持此语义：协议 token 原子化，但解码时保留协议文本。为统一 identity 管线，本方案把协议 token 都设为 `normalized=False`；这不保证复刻 DeepSeek 官方 `normalized=True` 行为或 ID。

协议 profile：默认 `qwen`；只需 DeepSeek 时用 `deepseek`；确需双协议时才用 `both`。`both` 只是两套集合的并集，不代表兼容任一厂商的原始 ID；要求逐 ID 对齐时直接加载官方 `tokenizer.json`。

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

保存为仓库外的 `~/min_tok.py`，替换语料路径后运行。脚本流式训练、注入完整 byte vocab、保存 tokenizer，并执行结构、ID、协议和无损门禁。

```python
import json
import re
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Sequence
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

CLASS_REGEX = (
    r"\p{White_Space}*"
    r"(?:"
    r"[\p{Han}\p{Hiragana}\p{Katakana}\u30FC]+"
    r"(?:[^\p{White_Space}\p{L}\p{N}]*[\p{Han}\p{Hiragana}\p{Katakana}\u30FC]+)*"
    r"|[\p{Latin}]+"
    r"(?:[^\p{White_Space}\p{L}\p{N}]*[\p{Latin}]+)*"
    r"|\p{N}+"
    r"(?:[^\p{White_Space}\p{L}]*\p{N}+)*"
    r")"
)

TOKEN_PROFILE = "qwen"  # 最小路径默认 Qwen；双协议部署改为 "both"

GENERIC_SPECIALS = ["<unk>", "<pad>", "<s>", "</s>"]

# Qwen3.8-27B：special=true 的真实 ChatML/多模态 token。
QWEN_SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<|audio_start|>",
    "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
    "<tts_text_bos_single>", "<|audio_pad|>",
]

# DeepSeek-V4.1 严格 profile：tokenizer.json 中的 special=true token。
# 不混入 V4 Pro 专有的 <｜image｜>；需要 V4 Pro 时另建 profile。
DEEPSEEK_SPECIALS = [
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>",
    "<｜end_of_query｜>", "<｜rl_image_pad｜>", "<｜rl_image_start｜>",
    "<｜/polygon｜>", "<｜polygon｜>", "<｜/point｜>", "<｜point｜>",
    "<｜/box｜>", "<｜box｜>", "<｜/ref｜>", "<｜ref｜>",
]

# Qwen 工具/协议 token：Qwen3.8 tokenizer.json 中 special=false。
QWEN_PROTOCOL_TOKENS = [
    "<\u200btool_call>", "</\u200btool_call>", "<tool_response>", "</tool_response>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<think>", "</think>",
]

# DeepSeek-V4.1 chat encoder 活跃 token：均为 tokenizer.json special=false。
DEEPSEEK_V4_1_CHAT_TOKENS = [
    "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<｜latest_reminder｜>",
    "｜DSML｜", "<｜deepseek_image｜>",
    "<｜action｜>", "<｜query｜>", "<｜authority｜>", "<｜domain｜>",
    "<｜title｜>", "<｜read_url｜>", "<think>", "</think>",
]

# 兼容、FIM 与 repo/file 扩展。旧 V4 tool control 仍存在，但不是 V4.1 DSML
# encoder 的必需控制流；按需保留，不能拿来替换 ｜DSML｜。
DEEPSEEK_COMPAT_PROTOCOL_TOKENS = [
    "<|EOT|>", "<dsml:", "</dsml:",
    "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>",
    "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>",
    "<｜tool▁outputs▁begin｜>", "<｜tool▁outputs▁end｜>",
    "<｜tool▁output▁begin｜>", "<｜tool▁output▁end｜>", "<｜tool▁sep｜>",
    "<｜fim▁hole｜>", "<｜fim▁begin｜>", "<｜fim▁end｜>",
    "<｜begin▁of▁repo▁name｜>", "<｜end▁of▁repo▁name｜>",
    "<｜begin▁of▁file▁name｜>", "<｜end▁of▁file▁name｜>",
    "<｜begin▁of▁file｜>", "<｜end▁of▁file｜>",
]
DEEPSEEK_PROTOCOL_TOKENS = [
    *DEEPSEEK_V4_1_CHAT_TOKENS,
    *DEEPSEEK_COMPAT_PROTOCOL_TOKENS,
]

if TOKEN_PROFILE == "qwen":
    SPECIALS = [*GENERIC_SPECIALS, *QWEN_SPECIALS]
    PROTOCOL_TOKENS = QWEN_PROTOCOL_TOKENS
elif TOKEN_PROFILE == "deepseek":  # 严格 DeepSeek-V4.1
    SPECIALS = [*GENERIC_SPECIALS, *DEEPSEEK_SPECIALS]
    PROTOCOL_TOKENS = DEEPSEEK_PROTOCOL_TOKENS
elif TOKEN_PROFILE == "both":  # 仅词表并集；chat renderer 仍须二选一
    SPECIALS = [*GENERIC_SPECIALS, *QWEN_SPECIALS, *DEEPSEEK_SPECIALS]
    PROTOCOL_TOKENS = [*QWEN_PROTOCOL_TOKENS, *DEEPSEEK_PROTOCOL_TOKENS]
else:
    raise ValueError(f"unknown TOKEN_PROFILE: {TOKEN_PROFILE}")

SPECIALS = list(dict.fromkeys(SPECIALS))
PROTOCOL_TOKENS = list(dict.fromkeys(PROTOCOL_TOKENS))
BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))

CORPUS_PATH = Path("/path/to/your/corpus.txt")
ALPHABET_PATH = Path("/path/to/your/alphabet.json")  # 可选；仓库样本见 dataset/pretok_cmp/alphabet.json


def iter_corpus(path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        yield from f


def build_initial_alphabet(corpus_path, alphabet_path=None):
    chars = {chr(cp) for cp in range(0x20, 0x7F)} | set("\n\r\t")
    for start, end in (
        (0x00A0, 0x0100), (0x0300, 0x0370), (0x2000, 0x2070),
        (0x3000, 0x3100), (0xFE00, 0xFE10), (0xFF01, 0xFF60),
    ):
        chars.update(chr(cp) for cp in range(start, end))
    if alphabet_path and alphabet_path.exists():
        chars.update(json.loads(alphabet_path.read_text(encoding="utf-8")))
    for line in iter_corpus(corpus_path):
        chars.update(line)
    return sorted(
        ch for ch in chars
        if len(ch) == 1
        and not 0xD800 <= ord(ch) <= 0xDFFF
        and not 0xFDD0 <= ord(ch) <= 0xFDEF
        and ord(ch) not in (0xFFFE, 0xFFFF)
    )


initial_alphabet = build_initial_alphabet(CORPUS_PATH, ALPHABET_PATH)
assert all(len(ch) == 1 for ch in initial_alphabet)
assert len(SPECIALS) == len(set(SPECIALS))
assert len(PROTOCOL_TOKENS) == len(set(PROTOCOL_TOKENS))
assert not set(SPECIALS).intersection(PROTOCOL_TOKENS)

# special token 在训练前注册；协议 token 训练后用 add_tokens 注册，保持 special=false。
tok = Tokenizer(BPE(byte_fallback=True, unk_token="<unk>"))
tok.normalizer = None
tok.pre_tokenizer = Split(Regex(CLASS_REGEX), behavior="merged_with_next")
tok.post_processor = None
for token in SPECIALS:
    tok.add_special_tokens([AddedToken(
        token, special=True, normalized=False, single_word=False,
        lstrip=False, rstrip=False,
    )])

trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=SPECIALS,
    initial_alphabet=initial_alphabet,
    show_progress=True,
)
tok.train_from_iterator(iter_corpus(CORPUS_PATH), trainer)

# Qwen/DeepSeek 的工具 token 实际是 special=false；不能用 add_special_tokens()。
for token in PROTOCOL_TOKENS:
    tok.add_tokens([AddedToken(
        token, special=False, normalized=False, single_word=False,
        lstrip=False, rstrip=False,
    )])


# byte_fallback 不会自动生成字节词表；训练后补齐全部 256 项。
def inject_byte_tokens(tokenizer):
    data = json.loads(tokenizer.to_str())
    model = data["model"]
    next_id = max(model["vocab"].values(), default=-1) + 1
    for token in BYTE_TOKENS:
        if token not in model["vocab"]:
            model["vocab"][token] = next_id
            next_id += 1
    model["byte_fallback"] = True
    model["unk_token"] = "<unk>"
    return Tokenizer.from_str(json.dumps(data, ensure_ascii=False))


tok = inject_byte_tokens(tok)
tok.decoder = Sequence([ByteFallback(), Fuse()])
tok.save("tokenizer.json")

reloaded = Tokenizer.from_file("tokenizer.json")
data = json.loads(reloaded.to_str())
assert data["normalizer"] is None
assert data["post_processor"] is None
assert data["model"]["byte_fallback"] is True
pretok = data["pre_tokenizer"]
assert pretok["type"] == "Split"
assert pretok["behavior"] == "MergedWithNext"
assert pretok["invert"] is False
assert "sentence_breaks" not in pretok
assert reloaded.token_to_id("<unk>") == 0
special_ids = [reloaded.token_to_id(token) for token in SPECIALS]
assert special_ids == list(range(len(SPECIALS)))
for token in SPECIALS:
    added = reloaded.get_added_tokens_decoder()[special_ids[SPECIALS.index(token)]]
    assert added.special is True and added.normalized is False
for token in PROTOCOL_TOKENS:
    token_id = reloaded.token_to_id(token)
    assert token_id is not None
    added = reloaded.get_added_tokens_decoder()[token_id]
    assert added.special is False and added.normalized is False
    assert not added.lstrip and not added.rstrip

byte_vocab = {
    token for token in reloaded.get_vocab()
    if re.fullmatch(r"<0x[0-9A-F]{2}>", token)
}
assert byte_vocab == set(BYTE_TOKENS)

sample = "Hello 你好，世界 123 abc"
enc = reloaded.encode(sample, add_special_tokens=False)
assert reloaded.decode(enc.ids, skip_special_tokens=False) == sample
assert enc.offsets[0][0] == 0 and enc.offsets[-1][1] == len(sample)
assert all(left[1] == right[0] for left, right in zip(enc.offsets, enc.offsets[1:]))
assert "".join(sample[start:end] for start, end in enc.offsets) == sample

split_cases = {
    "Hello你好": ["Hello", "你好"],
    "abc123你好456": ["abc", "123", "你好", "456"],
    "3.14": ["3.14"],
    "U.S.": ["U.S."],
    "a  b": ["a", "  b"],
}
for text, expected in split_cases.items():
    assert [part for part, _ in reloaded.pre_tokenizer.pre_tokenize_str(text)] == expected

# Qwen/DeepSeek 协议 token 必须原子匹配，且 skip_special_tokens=True 仍保留。
if TOKEN_PROFILE in ("qwen", "both"):
    qwen_tool_sample = (
        f"{QWEN_PROTOCOL_TOKENS[0]}get_weather{QWEN_PROTOCOL_TOKENS[1]}"
        f"{QWEN_PROTOCOL_TOKENS[2]}ok{QWEN_PROTOCOL_TOKENS[3]}"
    )
    qwen_enc = reloaded.encode(qwen_tool_sample, add_special_tokens=False)
    assert all(
        reloaded.token_to_id(token) in qwen_enc.ids
        for token in QWEN_PROTOCOL_TOKENS[:4]
    )
    assert reloaded.decode(qwen_enc.ids, skip_special_tokens=True) == qwen_tool_sample

if TOKEN_PROFILE in ("deepseek", "both"):
    # V4.1 使用 DSML；旧 <｜tool▁call▁begin｜> token 不是必需控制流。
    deepseek_tool_sample = (
        "<｜System｜>tools"
        "<｜Assistant｜><｜DSML｜ calls>"
        '<｜DSML｜ invoke name="get_weather">'
        '<｜DSML｜ parameter name="city" string="true">北京</｜DSML｜ parameter>'
        "</｜DSML｜ invoke>"
        "</｜DSML｜ calls>"
    )
    deepseek_enc = reloaded.encode(deepseek_tool_sample, add_special_tokens=False)
    assert all(
        reloaded.token_to_id(token) in deepseek_enc.ids
        for token in ("<｜System｜>", "<｜Assistant｜>", "｜DSML｜")
    )
    assert reloaded.decode(
        deepseek_enc.ids, skip_special_tokens=True
    ) == deepseek_tool_sample
```

运行：`cd ~ && python min_tok.py`。脚本末尾检查当前 `Split` 字段、固定 special ID、256 byte token、协议 flags/原子性、offset 连续性与 decode 无损。

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
