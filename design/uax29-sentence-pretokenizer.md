# C2 字符类 + Split 预分词（最终版）

**定案**：最终路径是 C2 `CLASS_REGEX` + 普通 `Split`，不启用句界层：

```python
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split
pretok = Split(Regex(CLASS_REGEX), behavior="merged_with_next", invert=False)
```

当前 Rust `Split` 只序列化 `pattern`、`behavior`、`invert`，没有 `sentence_breaks` 字段。C1 的句界实现只保留为历史 benchmark；C3 大正则只作性能参照。下方 Python API 以 `tokenizers==0.23.2` 验证路径为准；当前 checkout 的 Python binding 源码未导出这组预分词 API，默认构建也未提供该正则后端。

## 1. 最终 `CLASS_REGEX`

`behavior="merged_with_next"`、`invert=false`；最终正则如下：

```python
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
```

以上是相邻字符串拼接后的最终正则；换行和缩进只属于 Python 源码，不得进入 pattern。

三类显式 run：汉字/假名、拉丁、数字。JUNK 允许标点、组合符、ZWJ，并在同类再次出现时接回；类间不互串。两个约束不能改：汉字/拉丁分支的 JUNK 必须排除 `\p{N}`，否则 `abc123def` 会塌缩；不能用 `[\p{L}]` 兜底，否则汉字会进入 Latin run。数字分支刻意允许数字继续，因此 `3.14` 保持一个数字 run。

## 2. 行为

- 语言分离：`Hello你好→[Hello,你好]`；`abc123你好456→[abc,123,你好,456]`。
- 数字：`3.14→[3.14]`，不拆成 `[3,.,14]`；长数字串内部保持连续。
- 空白与尾 gap：`a  b→[a,  b]`；`hi `（末尾有空格）→`[hi ]`。`merged_with_next` 将间隔 gap 并入相邻片段，尾 gap 不会单独成片，原文不丢失。
- JUNK 拖尾：`U.S.`、`a.b` 保持拖尾并可续接同类；`1️⃣` 也可整体保留。
- 非显式脚本：泰老、阿拉伯、亚美尼亚、格鲁吉亚等不在三类显式分支中，按 gap 处理，可能并入相邻 match（`aกb→[aก,b]`），不保证独立切分。

## 3. JUNK 拖尾决策

数据来自 `dataset/pretok_cmp/results_cmp_junktrail.json` 与 `results_cmp_junktrail_effic.json`：

| 变体 | 预分词吞吐（1KB/100KB/1MB） | 片段/1k字符 | chars/token |
|---|---:|---:|---:|
| 保留拖尾（C2） | 15.25 / 10.14 / 8.78 | 133.0 | 3.1993 |
| 去掉拖尾 | 15.38 / 10.03 / 7.44 | 157.7 | 3.1524 |

**决策：保留 JUNK 拖尾组。** 压缩率约提升 1.5%，吞吐基本不变；去掉拖尾会增加碎片。两方案词表差异主要是标点附着方式。

## 4. 序列化与目标管线

| 组件 | 取值 |
|---|---|
| `normalizer` / `post_processor` | `None` |
| `pre_tokenizer` | `Split(CLASS_REGEX, merged_with_next, invert=false)` |
| `model` | `BPE(byte_fallback=True, unk_token="<unk>")` |
| `decoder` | `Sequence([ByteFallback(), Fuse()])` |

原文直通，模板交给调用方。`byte_fallback` 是推理期能力：训练器不会自动注入完整字节词表；需训练后注入 `<0x00>..<0xFF>`、设置 `model.byte_fallback=true`，再读回校验。保存后应检查关键字段，并验证训练语料 `encode→decode` 无损。

## 5. 限制与验证入口

限制：脚本白名单未闭合，未知脚本可能被 Latin run 吸入；`U.S.A.`、`left.Now` 与小数点存在同形歧义，无句界层时不建模，由 Model 处理。升级 `tokenizers`、Unicode 数据或正则引擎后需重跑行为门禁。

可执行入口：

- `design/minimal-tokenizer-path.md`：`tokenizers==0.23.2` 的独立 PyPI 脚本；替换语料路径后运行，检查保存、读回、256 个 byte token 和无损往返。预分词行为以本文无前导换行的正则定义为准。
- `dataset/pretok_cmp/cmp_junktrail.py`：`effic` / `report` 对照；直接加载固定路径的 `tokenizers.abi3.so`，依赖旧编译 binding，需先校准路径。
- 在 `tokenizers/` 运行 `cargo test -p tk-encode split`：当前 Rust 通用 `Split` 回归；当前源码没有 `CLASS_REGEX` 专用单测。

行为门禁至少覆盖：`Hello你好`、`abc123你好456`、`3.14`、`U.S.`、`a.b`、`a  b`、`hi `、`aกb`、保存读回及 `encode→decode` 无损。
