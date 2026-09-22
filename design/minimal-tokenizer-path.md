# 自研 Tokenizer 最小路径：实施步骤

> 本文件将 `tokenizer-architecture.md` §9「自研 tokenizer 准备：最小路径建议」落为**可执行步骤**。
> 核心约束：**不修改仓库源码**——全程用 PyPI 发布的 `tokenizers` 包，不构建本地 Rust core、不改动任何源文件。脚本放在仓库外运行即可。
>
> 配套设计：`uax29-sentence-pretokenizer.md`（预分词正则与行为）、`pretok-performance.md`（性能基准）。

---

## 0. 目标管线（一图速览）

| 组件 | 取值 | 说明 |
|---|---|---|
| 特殊 token | `AddedToken(t, special=True)` 提前注册 | 在 normalizer 之前、按原文切分 |
| `normalizer` | `None` | 原文直通，大小写/重音/全角/空白/控制字符全保留 |
| `pre_tokenizer` | `Split(CLASS_REGEX, merged_with_next)` | 字符类切分（无句界） |
| `model` | `BPE(byte_fallback=True, unk_token="<unk>")` | 字节回退兜底（推理期） |
| `post_processor` | `None` | 先跑通全链，再按需加 bos/eos |
| `decoder` | `Sequence([ByteFallback(), Fuse()])` | 推理期还原 |

---

## 1. 环境准备（一次性）

```bash
# 用发布的 tokenizers，而非本地仓库构建 → 天然"不修改仓库代码"
python -m venv ~/min_tok.venv && source ~/min_tok.venv/bin/activate
pip install "tokenizers>=0.21"   # 需支持 byte_fallback / Split(behavior)
```

> 仓库的 `design/` 文档描述的是**源码内部行为**（`Split` 的 `sentence_breaks`、Rust 单测等）。本步骤只消费其结论（CLASS_REGEX 字符串 + 管线取值），不依赖仓库源码。若想复现 `sentence_breaks` 句界扩展，才需要触碰 `tokenizers/src/pre_tokenizers/split.rs` 等源码，本最小路径不需要。

---

## 2. 实施步骤（对应 §9 五点）

| §9 条目 | 落地动作 | Python 取值 |
|---|---|---|
| 抽取特殊 token | `add_special_tokens` 在训练前注册（在 normalizer 之前、按原文匹配） | `AddedToken(t, special=True)` |
| 规范化 minimal/identity | 显式置 `None`，原文直通 | `tok.normalizer = None` |
| 预分词 | 字符类 `Split`（无句界） | `Split(Regex(CLASS_REGEX), "merged_with_next")` |
| 分词模型 | BPE + byte_fallback | `BPE(byte_fallback=True, unk_token="<unk>")` |
| post_processor | 先 `None`，跑通全链再按需加 | `tok.post_processor = None` |

### 2.1 完整训练脚本（保存为 `~/min_tok.py`，**放在仓库外**）

```python
from tokenizers import Tokenizer, AddedToken, Regex
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

# ── 2. 特殊 token（提前注册；normalizer=None 时按原文切分）──
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]

# ── 3. 组装最小管线 ──
tok = Tokenizer(BPE(byte_fallback=True, unk_token="<unk>"))
tok.normalizer = None                                   # 原文直通
tok.pre_tokenizer = Split(Regex(CLASS_REGEX), behavior="merged_with_next")
tok.post_processor = None                               # 教学原型先 null
for s in SPECIALS:
    tok.add_special_tokens([AddedToken(s, special=True)])

# ── 4. 训练器：byte_fallback 仅推理期生效 → 覆盖率只能靠 initial_alphabet ──
initial_alphabet = sorted(set(chr(c) for c in range(0x20, 0x7F)))  # 至少覆盖 ASCII
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=SPECIALS,
    initial_alphabet=initial_alphabet,   # 真实场景用全语料字符集替换
    show_progress=True,
)

# ── 5. 流式训练（避免整份语料常驻 OOM）──
def corpus():
    with open("/path/to/your/corpus.txt", encoding="utf-8") as f:
        for line in f:
            yield line

tok.train_from_iterator(corpus(), trainer)

# ── 6. 解码器：推理期还原 ──
tok.decoder = Sequence([ByteFallback(), Fuse()])

# ── 7. 落盘 + 往返无损断言（§9 要求）──
tok.save("tokenizer.json")
reloaded = Tokenizer.from_file("tokenizer.json")
import json
j = json.loads(reloaded.to_str())
assert j["normalizer"] is None, "normalizer 必须是 null"
assert j["post_processor"] is None, "post_processor 必须是 null"
assert j["model"]["byte_fallback"] is True, "byte_fallback 必须为 True"

sample = "Hello 你好，世界 123 abc"
enc = reloaded.encode(sample)
assert enc.decode() == sample, "encode→decode 必须无损"

# ── 8. （可选第二步）指令微调自动加尾词 ──
from tokenizers.processors import TemplateProcessing
reloaded.post_processor = TemplateProcessing(
    single="$0 </s>",
    special_tokens=[("</s>", reloaded.token_to_id("</s>"))],
)
```

### 2.2 验证命令

```bash
source ~/min_tok.venv/bin/activate
python ~/min_tok.py                      # 落盘 tokenizer.json + 三个断言 + 无损断言
python - <<'PY'                          # 独立复验序列化往返
import json
j = json.load(open("tokenizer.json"))
print(j["normalizer"], j["post_processor"], j["model"]["byte_fallback"])
PY
```

---

## 3. 关键坑位（来自设计文档 §6 / §9）

1. **`byte_fallback` 是推理期属性**：`BpeTrainer` 不接受该选项，`initial_alphabet` 只取每项首字符，无法注入 256 个 `<0xXX>`。所以训练期覆盖率**只能靠 `initial_alphabet = 全语料字符集`**。若需真正覆盖任意字节，训练后做 `tokenizer.json` 手术：注入 `<0x00>`..`<0xFF>` 并设置 `model.byte_fallback=true`，再 `from_file` 重建。
2. **`unk_token="<unk>"` 兜底**：避免 `unk_token=None` 且字符既非词汇又无 `<0xXX>` 时被静默丢弃。
3. **`post_processor` 先 `None`**：`ByteLevel(trim_offsets=true)` / `RobertaProcessing` 会改 offsets，破坏 UAX29 无损约定；`TemplateProcessing/BertProcessing` 会自动加 CLS/SEP，与「手动控制」冲突。先跑通 `raw→split→BPE→Encoding→decode` 全链 + offset 单测，再按需加 eos。
4. **`merged_with_next` 的合并方向**：每个 match 与**后置** gap 合并（前置 gap 独立 piece），如 `Hello, world! → ['Hello,', ' world!']`。断言不要用 span 直接比对，改用「拼接==输入 + 去空白后核心 token 顺序」或「offset 连续单调」。
5. **流式训练防 OOM**：整份语料常驻 + trainer 词频表曾触发 OOM；务必 `train_from_iterator` 逐行，`initial_alphabet` 从语料统计得到后传入。

---

## 4. 后续（按需）

- 需要句界隔离时，开启 `sentence_breaks=True`（仅仓库源码支持，见 `uax29-sentence-pretokenizer.md` §10）。
- 性能基准参考 `pretok-performance.md`；编码成本主要在预分词阶段（C2 约 11/6.5/4.7 MiB/s，release 构建）。
- 升级 `tokenizers` / 换正则引擎时，重跑 §2.2 的序列化与无损断言门禁。

---

## 5. 复用现有字符级 BPE 词表（免重训）

> 对 23 个主流分词器（`design/tokenizer-landscape-survey.md`）的核查结论：**全部是 byte-level 实现**——`pre_tokenizer` 尾部必为 `ByteLevel`，词表是 byte→unicode 映射（`Ġ`/`Ċ` 等），与本方案的字符级偏好不符。
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

### 5.2 选型

- **中文优先（CJK 导向，契合 CLASS_REGEX）** → `baichuan2`：20,338 个单字汉字、含汉字 piece 占 56%，中文无损率最好、词表最大。代价：只有 SPM `tokenizer.model`，需 `sentencepiece`（已装）或 `transformers` 加载，并一次性转成 `tokenizer.json`。
- **免转换 + 宽松许可** → `yi`（Apache-2.0，63,992）或 `internlm2`（92,544），二者均带 `tokenizer.json`，`Tokenizer.from_file` 直接可用。
- 三者构件均满足 `model = BPE(byte_fallback=True)`，与 §0/§2 的 `model` 取值**字面一致**——**无需 §3 坑位①的「注入 `<0xXX>` + 置 byte_fallback」手术**。

### 5.3 复用步骤（免重训，只换管线）

1. 载入词表（`baichuan2` 走 SPM；`internlm2`/`yi` 走 `Tokenizer.from_file("tokenizer.json")`）。
2. `tok.normalizer = None`。SPM 系把空格烧成 `▁`，须一并去掉（见 §5.4 坑位④）。
3. `tok.pre_tokenizer = Split(Regex(CLASS_REGEX), behavior="merged_with_next")` 替换原 `null`/`Metaspace`。
4. **`▁` 重写**：SPM 词表 piece 带 `▁` 前缀（如 `▁你好`），而 `normalizer=None` 下输入是真实空格，匹配不上 `▁` piece。必须把 key 做 **`▁X → " X"`** 重写（或保留 `Replace(" "→"▁")`，但那就偏离 `normalizer=None`）。
5. 保留 `byte_fallback=True`、`unk_token="<unk>"`，`decoder = Sequence([ByteFallback(), Fuse()])`。
6. 落盘后跑 §2.1 的「序列化三断言 + 无损断言」门禁。

### 5.4 关键坑位（承接 §3）

4. **`▁` 与 `normalizer=None` 冲突**：SPM 系字符表以 `▁` 标记词边界，与本方案 `normalizer=None`（空格原样）不兼容。拿到词表后必须 `▁X → " X"` 重写 piece key，否则中/英文 token 全部失配。

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

old = Tokenizer.from_file("dataset/pretok_cmp/tokenizer_C2_nosb_sameclass.json")
chars = set()
for tok_str in old.get_vocab():        # get_vocab() -> {token: id}
    chars.update(tok_str)              # 逐字符拆开
for line in new_corpus():
    chars.update(line)                 # 并入新语料字符
initial_alphabet = sorted(chars)       # = 定义好的「初始词表」（字符级）
```

> 实仓写法见 `dataset/pretok_cmp/train_and_compress.py` 的 `cmd_train`：`alpha = set(); for line in iter_lines(): alpha.update(line)`，再 `initial_alphabet=sorted(alpha)`。逻辑完全一致，只是把来源从「纯语料」换成「旧 vocab + 新语料」。

### 6.3 使用：种子注入 + 续训

```python
from tokenizers import Tokenizer, Regex, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import ByteFallback, Fuse, Sequence

tok = Tokenizer(BPE(byte_fallback=True, unk_token="<unk>"))
tok.normalizer = None
tok.pre_tokenizer = Split(Regex(CLASS_REGEX), behavior="merged_with_next")
tok.post_processor = None
tok.decoder = Sequence([ByteFallback(), Fuse()])

trainer = BpeTrainer(
    vocab_size=32000,                  # 比旧词表略大，给新 token 留空间
    min_frequency=2,
    initial_alphabet=initial_alphabet, # ← §6.2 定义的初始词表（字符级）
    special_tokens=["<unk>"],          # 旧特殊 token 一并带上
    show_progress=True,
)
# 在「新语料」上训练；若想保留旧 token 词频，把旧语料也并入迭代器
tok.train_from_iterator(new_corpus(), trainer)
tok.save("tokenizer_extended.json")
```

### 6.4 坑位（承接 §3）

1. **旧多字符 token 不自动保留**：`initial_alphabet` 只保「字符」，不保「词」。旧 token（如 `你好`）只有当新语料里仍以足够频率出现时才会被重新学出。要兼顾旧词频，把**旧语料也喂进** `train_from_iterator` 的迭代器（与 `train_and_compress.py` 的 `iter_lines()` 同口径）。
2. **字节回退 token 无法注入**：`<0xXX>` 是 6 字符串，不能用 `initial_alphabet` 注入；训练后仍需做 `tokenizer.json` 手术（注入 `<0x00>`..`<0xFF>` 并置 `model.byte_fallback=true`，见 §3 坑位①），否则未见字节仍走 `<unk>`。
3. **vocab_size 预留余量**：扩展训练词表通常比旧词表大，设 `vocab_size` 时给新 token 留空间，否则新语料的高频子词会被截断。
4. **非破坏性替代见 §6.5**：若只想新增少量领域词/专名、保旧词表稳定，改用 `add_tokens()`，不重算合并。

### 6.5 对照：仅追加 token（方案 B，非破坏性）

```python
tok = Tokenizer.from_file("dataset/pretok_cmp/tokenizer_C2_nosb_sameclass.json")
added = tok.add_tokens(["新词1", "专用术语XYZ"])   # 返回实际新增条数
print("新增:", added, "|V|:", tok.get_vocab_size())
tok.save("tokenizer_extended.json")
```

代价：新 token 为整块单 token，不参与子词合并，压缩率略吃亏，但覆盖与语义确定；旧词表原样保留。

### 6.6 选型小结

| 目标 | 方案 | 出处 |
|---|---|---|
| 新语料很大，重学最优合并、扩充词表 | **A**（种子注入 + 续训，旧语料并入） | §6 |
| 仅加少量领域词/专名，旧词表稳定 | **B**（`add_tokens`） | §6.5 |
| 直接复用现成词表、只换管线 | 免重训复用 | §5 |
