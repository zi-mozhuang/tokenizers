# 设计方案 v3（定稿）：SentenceSplit —— 基于 Unicode UAX #29 的无损句子预分词器

> v1：ICU4X 段落+句子两级方案（复杂）
> v2：正则组合主路径 + 兜底模块（正则保真度不足，否决）
> **v3（本版）**：单模块 `SentenceSplit`，直接输出 UAX #29 默认（root）句子分割，
> 遵循标准的一切默认行为，不做任何自定义裁剪。核心代码约 15 行。

---

## 1. 设计原则（硬约束）

| # | 约束 | 说明 |
|---|------|------|
| 1 | **无损** | 不增、不删、不改原文任何字符；所有 splits 拼接 == 原文 |
| 2 | **遵循标准默认** | 边界位置、空白归属完全按 UAX #29 root rules，不做 tailoring |
| 3 | **无段落层级** | UAX #29 本身不定义 paragraph，本模块也不引入 |
| 4 | **最简化** | 零配置项、零新增依赖、单模块单节点 |

## 2. UAX #29 标准行为（本模块 = 标准行为的直接映射）

- **SB1/SB2**：文本首尾为天然边界；
- **SB3**：`CR × LF` —— CRLF 视为一个单元，不拆；
- **SB4**：每个 `Sep | CR | LF` 之后都断 —— **每个换行符自成一个 split**（无"空行才是段落"概念）；
- **SB5**：`× Extend | Format | ZWJ` —— 组合字符、ZWJ emoji、国旗序列永远不切在中间；
- **SB6–SB10**：缩写与数字上下文保护 —— `Mr.`、`e.g.`、`D.C.`、`3.14` 等**不切**；
- **SB8a/SB9**：引号、括号等 Close 字符附着在其闭合的句子上 —— `「こんにちは。」` 是一个整体；
- **SB11**：主断句规则，边界在 `SATerm Close* Sp*` **之后** —— **句间空白归入前一句末尾**。

> 关键澄清：标准只定义边界位置（÷），不存在"空白独立成 token"或"并入后句"的概念；
> MergedWithPrevious/Next/Isolated 是 tokenizers 框架的实现层概念，本模块不需要它们。
> 有损行为（丢空白、替换 ▁、字节映射）是现有部分模块的工程选择，与本模块无关。

**已知边界（标准自身的限制，不是实现缺陷）**：
- 无 locale tailoring（希伯来文缩写等）——root rules 是 locale 无关默认规则；
- 泰文等无空格语言的隐式句界**不在标准范围内**（词典模型只用于词界，不用于句界）；
- 换行符各自成段（SB4），硬换行散文会在每个 `\n` 处断开——这是标准行为，有意保留。

## 3. 模块设计

### 3.1 依赖

`unicode-segmentation = "1.11"` —— **已在 `tokenizers/Cargo.toml` 依赖树中**，零新增依赖、
无需 feature 门控。其 `split_sentence_bound_indices` 是 UAX #29 root rules 的完整
状态机实现，对官方 `SentenceBreakTest.txt` 一致性为 100%。

### 3.2 核心实现（`tokenizers/src/pre_tokenizers/sentence.rs`，新文件）

```rust
use crate::tokenizer::{normalizer::Range, PreTokenizedString, PreTokenizer, Result};
use crate::utils::macro_rules_attribute;
use unicode_segmentation::UnicodeSegmentation;

/// 基于 Unicode UAX #29 默认句子边界的无损预分词器。
/// - 不增删改任何字符（splits 拼接 == 原文）
/// - SB11：句间空白归入前句末尾；SB4：每个换行符自成 split
/// - 缩写、引号、组合字符、ZWJ emoji 永不切开（SB5-SB10）
#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct SentenceSplit;

impl SentenceSplit {
    pub fn new() -> Self { Self {} }
}

impl Default for SentenceSplit {
    fn default() -> Self { Self::new() }
}

impl PreTokenizer for SentenceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            let mut offsets: Vec<usize> = normalized
                .get()
                .split_sentence_bound_indices()
                .map(|(i, _)| i)
                .collect();
            offsets.push(normalized.get().len());
            Ok(offsets
                .windows(2)
                .map(|item| {
                    normalized
                        .slice(Range::Normalized(item[0]..item[1]))
                        .expect("NormalizedString bad split")
                })
                .collect::<Vec<_>>())
        })
    }
}
```

要点：
- 复用与 `UnicodeScripts` 完全相同的切片机制（`pretokenized.split()` +
  `Range::Normalized`），offsets 映射（normalized → original）由框架自动维护；
- `split_sentence_bound_indices` 产出 grapheme 对齐的字节偏移，零长度片段不存在，
  区间并集 == 输入 → 无损由构造保证；
- JSON 序列化最简形态：`{"type":"SentenceSplit"}`。

### 3.3 注册点（对照 `FixedLength`/`UnicodeScripts` 的既有改法）

**Rust core — `tokenizers/src/pre_tokenizers/mod.rs`，6 处：**
1. `pub mod sentence;`
2. `use crate::pre_tokenizers::sentence::SentenceSplit;`
3. `PreTokenizerWrapper` 枚举 + `pre_tokenize` match 分支；
4. 反序列化 `EnumType` 加 `SentenceSplit` + 对应 match 分支；
5. `PreTokenizerUntagged` 加变体 + 对应 match 分支；
6. `impl_enum_from!(SentenceSplit, PreTokenizerWrapper, SentenceSplit);`

**Python — `bindings/python/src/pre_tokenizers.rs`，4 处：**
1. `use tk::pre_tokenizers::sentence::SentenceSplit;`
2. `get_as_subtype` 加 `PreTokenizerWrapper::SentenceSplit(_) => Py::new(py, (PySentenceSplit {}, base))` 分支；
3. 仿 `PyUnicodeScripts`（L879-889）新增 `PySentenceSplit` pyclass（无构造参数）；
4. `#[pymodule_export] pub use super::PySentenceSplit;`
   stub（`.pyi`）由 `make check-style` 自动生成，**不手改**。

**Node — `bindings/node/src/pre_tokenizers.rs`，1 处：**
仿 `whitespace_split_pre_tokenizer()`（L131-138）新增工厂函数：
```rust
#[napi]
pub fn sentence_split_pre_tokenizer() -> PreTokenizer {
  PreTokenizer {
    pretok: Some(Arc::new(RwLock::new(
      tk::pre_tokenizers::sentence::SentenceSplit.into(),
    ))),
  }
}
```
（注：Node 侧本来就未导出全部 12 个模块——`UnicodeScripts`/`FixedLength` 也缺；
导出 SentenceSplit 属可选增强，不做也不影响 core/Python。）

## 4. 边界行为速查表

| 输入 | 输出 splits | 规则依据 |
|------|-------------|----------|
| `Mr. Smith went.  He left.` | `["Mr. Smith went.  ", "He left."]` | SB6-8 不切缩写；SB11 空白归前句 |
| `他说：你好。转身走了。` | `["他说：你好。", "转身走了。"]` | 。 为 STerm |
| `彼は言った。「こんにちは。」そして去った。` | `["彼は言った。", "「こんにちは。」", "そして去った。"]` | SB8a/SB9 引号附着 |
| `a\r\nb\n\nc` | `["a", "\r\n", "b", "\n", "\n", "c"]` | SB3 CRLF 一体；SB4 每个换行自成 split |
| `Café 👨‍👩‍👧 done.` | 单 split，组合字符/ZWJ emoji 不切 | SB5 |
| `3.14 is pi. 1.5.2 is not.` | `["3.14 is pi. ", "1.5.2 is not."]` | SB6/SB8 数字上下文 |
| `Wait… what?! Really?!…` | `["Wait… ", "what?! ", "Really?!…"]` | SB9/SB11 终止符序列归并 |
| `"末尾无终止符"` | `["末尾无终止符"]` | SB2 ÷ eot |

## 5. 测试方案

1. **无损 property test**（core 单元测试，随模块同文件）：空串、纯空白、文末终止符、
   连续换行、组合字符/ZWJ emoji、多语言混排 → 断言 splits 拼接 == 原文
   且 offsets 连续覆盖 `[0, len)`；
2. **多语言 golden**：中/英/日/韩/藏(།)/印地(।)/法(NBSP)/阿拉伯(؟)/CRLF（见 §4 表）；
3. **官方一致性**（可选增强）：`make data/SentenceBreakTest.txt` 下载官方用例后
   `cargo test --test sentence_split` 逐条断言（`unicode-segmentation` 本身已 100% 通过，
   该测试验证的是集成层不引入偏差）；
4. **Python 测试**：`tests/bindings/test_pre_tokenizers.py` 加 JSON 序列化往返
   （`__getstate__/__setstate__`）与 `pre_tokenize_str` 行为断言。

## 6. 与现有 12 模块的关系

本模块成为第 13 个 pre-tokenizer。无损性审计（哪些可与之组合）：

| 可无损组合 | 有损（不可用于本约束） |
|------------|------------------------|
| Split（非 Removed）、Punctuation（非 Removed）、Digits、UnicodeScripts、FixedLength、Sequence | Whitespace、WhitespaceSplit、BertPreTokenizer、CharDelimiterSplit（丢字符）；ByteLevel、Metaspace（改写字符） |

典型组合（句内细分，SentenceSplit 先行形成句界 barrier，后续模块只在句内运行）：

```python
from tokenizers import pre_tokenizers
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.SentenceSplit(),      # 1. UAX #29 句界（无损）
    pre_tokenizers.UnicodeScripts(),     # 2. 句内按 Script 细分（无损）
    pre_tokenizers.Digits(False),        # 3. 句内数字段（无损）
])
```

## 7. 实施步骤

1. 新建 `tokenizers/src/pre_tokenizers/sentence.rs`（§3.2 代码 + §5.1/5.2 单元测试）；
2. `tokenizers/src/pre_tokenizers/mod.rs` 注册（§3.3 Rust core 6 处）；
3. WSL 内验证：`cd tokenizers && cargo test --lib sentence` 和
   `cargo clippy --all-targets --all-features -- -D warnings`；
4. Python binding（§3.3 Python 4 处）→ `maturin develop` →
   `python -m pytest tests/bindings/test_pre_tokenizers.py -v -k sentence`；
5. `make check-style` 重新生成 stub 并检查 diff；
6. Node binding（§3.3 Node 1 处，可选）→ `yarn build && make test`；
7. 文档：两套独立来源 `docs/source/`（Sphinx）与 `docs/source-doc-builder/` 需分别更新。
