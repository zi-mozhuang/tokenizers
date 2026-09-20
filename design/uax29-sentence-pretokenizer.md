# 设计方案：UAX #29 句界 + 语言/数字/空白分类的无损预分词（单 `Split` 节点）

> **目标**：单节点完成 ① UAX #29 句界 ② 语言分离 ③ 数字独立且内部连续 ④ 空白不删、附着后面。
> **硬约束**：不增删改任何字符（splits 拼接 == 输入，offsets 无损回原文）。
> **结论**：`Split` 加可选布尔字段 `sentence_breaks`，内部走 `PreTokenizedString::split` 的**递归通道**——先在句界切片，再在每句内跑 `CLASS_REGEX` + `MergedWithNext`。
> 句界因此是结构性边界，逐例等价于两级 `Sequence([SentenceSplit, Split])`，UAX #29 句界零损失。全部切点由 `slice(Range::Normalized(a..b))` 构造，无损性由构造保证。

## 0. 关键决策

| 项 | 决策 |
|----|------|
| 句界承载 | `Split` 内部两级递归；不用正则表达、不揉进 run 列表 |
| 句界来源 | `unicode-segmentation::split_sentence_bound_indices`（SB1–SB16） |
| 句界空白 | 并入前句（SB11），由内层 gap 朝前粘自然满足 |
| 空白 | 不替换，写进 match 前缀；**不得**拆成独立 `WS+` 分支（会破坏 SB11 形态） |
| 标点/组合符/emoji | 默认归前段（`merged_with_next`），与 SB9 方向一致 |
| 数字 | 独立类、内部连续；数字内标点（`3.14`）是自由选项，两种都无损 |
| 脚本 | 按 Script 变化切分；白名单**可闭合**（偏差 ④） |
| 段落层级 | 放弃（UAX #29 无 paragraph） |
| 泰文/高棉文隐式句界 | 不在范围（标准的词典模型只用于 word 界） |

**已否决**：① ICU4X（依赖链 + feature 会改 `PreTokenizerWrapper` 的 serde 形状）；② 用一条正则表达句界（SB8 需无界前瞻、SB8a–SB11 有状态，且 `U.S.A.` 与 `left.Now` 字符级同形）；③ 把句界揉进 run 列表（见 §1）；④ 独立 `ClassSplit` 模块（一条正则已等价）；⑤ 两级 `Sequence`（等价，仅多一个节点，保留作等价性测试参照）。

> **代价**：BERT 的 `BertPreTokenizer`、GPT-2 的 `ByteLevel` 在无损约束下都不可复刻；本管线定位为字符级 / Unigram / CJK 优先、需保留原文 offsets 的场景。

## 1. 为什么句界必须放在递归层

`Pattern::find_matches` 产出 run 列表（`pattern.rs:63-83`），`NormalizedString::split` 按 behavior 对**整份列表**折叠（`normalizer.rs:694-783`）：`MergedWithNext` 让 match 吸收右侧 gap，`MergedWithPrevious` 吸收左侧 gap，`Contiguous` 合并同 flag 邻 run，`Removed` 丢 match。故"pattern 内部构造的边界"可能被跨越。

而 `PreTokenizedString::split`（`tokenizer/pre_tokenizer.rs:73-103`）逐层递归，**层间边界物理上不可跨越**。

```
"a\r\nb\n\nc"      UAX 句界 {1,3,4,5,6}
  run 列表融合  →  "a\r\n" | "b\n" | "\n" | "c"        边界 {3,5,6}   ← 丢 1、4
  逐句递归      →  ["a"]["\r\n"]["b"]["\n"]["\n"]["c"]  全保住
```

原因：句子 `"\r\n"` 只有一个 gap run，`MergedWithNext` 让前句末尾的 match 把它吸走；整句无 match 可"提升"，无法打补丁（谎报 flag 需同时禁 4 种 behavior，不值得）。分句口径按 `sentence.rs:110-124`，**须先实测定口径**。

**句内**仍靠 gap 朝前粘：`abc!你好` → match `abc` 吸收 gap `!`，段界落在每个 match 起点；`abc 你好` → 空格写进 match 前缀，随 match 落到后一段。于是 ①③ 由"脚本分支 + 空白前缀"实现，② 由数字独立分支实现，标点归前段自动成立。

不用 `UnicodeScripts`/`Sequence` 拼装：前者把空格映射为 `Any` 并恒吸入**前** run（与规则 ③ 相反）；后者存在下游对纯空白 split 的 `windows(2)` 为空导致的数据丢失路径。

## 2. 内层 pattern

```json
{"type": "Split",
 "pattern": {"Regex": "\\p{White_Space}*(?:[\\p{Han}\\p{Hiragana}\\p{Katakana}\\u30FC]+(?:[^\\p{White_Space}\\p{L}\\p{N}]*[\\p{Han}\\p{Hiragana}\\p{Katakana}\\u30FC]+)*|[\\p{Latin}]+(?:[^\\p{White_Space}\\p{L}\\p{N}]*[\\p{Latin}]+)*|\\p{N}+(?:[^\\p{White_Space}\\p{L}]*\\p{N}+)*)"},
 "behavior": "MergedWithNext", "invert": false, "sentence_breaks": true}
```

```
CLASS_REGEX = \p{White_Space}* (?: X_HAN | X_LATIN | X_NUM )
X_HAN   = [\p{Han}\p{Hiragana}\p{Katakana}\u30FC]+ (?: JUNK* [Han/Kana/\u30FC]+ )*
X_LATIN = [\p{Latin}]+                            (?: JUNK* [\p{Latin}]+ )*
X_NUM   = \p{N}+ (?: [^\p{White_Space}\p{L}]* \p{N}+ )*
JUNK    = [^\p{White_Space}\p{L}\p{N}]
```

- `\p{White_Space}*` = 规则 ③；引擎不支持该属性名时用显式类 `[\t\n\r\f\v\x{20}\x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]`。
- `JUNK` ≈ `Script::Common`/符号/emoji 的"非空白非字母非数字"刻画；`(?:JUNK* S+)*` 让同类 run 内的标点/组合符被吸收。
- **两个必须写死的细节**：`X_LATIN`/`X_HAN` 的 JUNK 要排除 `\p{N}`（否则 `abc123def` 被吞成单 match）；不得用 `[\p{L}]` 兜底（否则汉字被吞进拉丁 run）。

代表用例（同时是须对外公布的偏差清单）：

| 输入 | 输出 | 备注 |
|---|---|---|
| `价格是 100 元` / `Hello world` | `["价格是"," 100"," 元"]` / `["Hello"," world"]` | ③+①② |
| `3.14 and 1.5.2` / `1,000元` | `["3.14"," and"," 1.5.2"]` / `["1,000","元"]` | ② |
| `你好。abc` / `他说：你好` / `どこで生れ` | `["你好。","abc"]` / 单段 / 单段 | 标点归前段；Common 吸收；kana→Han |
| `abc  ` | `["abc  "]` | 偏差 ② 末尾空白并回前段（= SB11） |
| `。abc` | `["。","abc"]` | 偏差 ① 句内作用域首部自成一段；与 `abc。你好` 不可兼得 |
| `see 👨‍👩‍👧 done` | `["see 👨‍👩‍👧"," done"]` | 偏差 ③ 空白跟非字母内容时朝前 |
| 泰文段+天城文段相邻 | 单段 | 偏差 ④ 白名单；可闭合（135 分支，约 4.8k 字符，损失 ~20% 吞吐 + `sync_check.py`） |

## 3. 实现

`tokenizers/src/pre_tokenizers/split.rs`：

1. `Split` 加 `#[serde(default, skip_serializing_if = "is_false")] pub sentence_breaks: bool`（`fn is_false(b: &bool) -> bool { !*b }`）；`SplitHelper` 加同名字段（旧 JSON 仍可反序列化）；`Split::new` 签名不变；`Clone`（`:61-65`）/`PartialEq`（`:67-73`）带上该字段。
2. `pre_tokenize` 分流 + 新增 `sentence_aware_segments`。

```rust
use crate::tokenizer::normalizer::Range;
use crate::tokenizer::NormalizedString;
use unicode_segmentation::UnicodeSegmentation;

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        if self.sentence_breaks {
            return pretokenized.split(|_, normalized| {
                sentence_aware_segments(&normalized, &self.regex, self.behavior)
            });
        }
        if self.invert {
            pretokenized.split(|_, n| n.split(Invert(&self.regex), self.behavior))
        } else {
            pretokenized.split(|_, n| n.split(&self.regex, self.behavior))
        }
    }
}

/// 外层 UAX #29 句界；内层每句跑原 pattern + behavior
fn sentence_aware_segments(
    normalized: &NormalizedString,
    regex: &SysRegex,
    behavior: SplitDelimiterBehavior,
) -> Result<Vec<NormalizedString>> {
    let mut starts: Vec<usize> =
        normalized.get().split_sentence_bound_indices().map(|(i, _)| i).collect();
    starts.push(normalized.get().len());
    starts.dedup(); // 防御零长句 → 空 split

    let mut splits = Vec::with_capacity(starts.len());
    for w in starts.windows(2) {
        if w[0] == w[1] { continue; }
        let sentence = normalized.slice(Range::Normalized(w[0]..w[1]))?;
        splits.extend(sentence.split(regex, behavior)?);
    }
    Ok(splits)
}
```

`sentence.rs` 也应同步补 `dedup`。不触碰 `normalizer.rs`，不新增 `Pattern`，不改 `SplitPattern`。

- **等价性**：与 `Sequence([SentenceSplit, Split])` 的句子区间、内层调用序列、slice 对齐完全一致 ⇒ 含 offsets 逐例相等。收益只是少一个 JSON 节点；性能收益为个位数百分比。
- **约束**：`behavior` 五种皆可用（`Removed` 与无损前提冲突）；`invert` = 句内取反；`sentence_breaks=false` 零回归；`pattern` 作用域收窄为句内，须写入文档。
- 句界在 `PreTokenizedString` 中间态不可见；需要"句子"当一等片段时用 §4 的可选注册。

## 4. 绑定

`PySplit::new`（`bindings/python/src/pre_tokenizers.rs:420-433`）加参数，同步 getter/setter 与 docstring：

```rust
#[pyo3(signature = (pattern, behavior, invert = false, sentence_breaks = false),
      text_signature = "(self, pattern, behavior, invert=False, sentence_breaks=False)")]
```

Node 加工厂参数并同步 `lib/` 的 TS 声明。**不要手改生成的 `.pyi`**（`make check-style` 会重生成）。测试放 `bindings/python/tests/bindings/test_pre_tokenizers.py`（JSON 往返 + 旧配置兼容 + 行为断言）。建议 binding 侧把 `CLASS_REGEX` 固化为常量。

可选：`sentence.rs` 已实现但**未注册**（`mod.rs:1-11` 与 `PreTokenizerWrapper:30-43` 均缺），只需单节点时不必注册；需要句子一等片段时再补 `mod.rs` 7 处 + Python 4 处。

## 5. 测试

1. **无损 property**：拼接 == 输入；两个 referential 的 offsets 连续、单调、无重叠、并集 == 全串；无空 split。覆盖空串/纯空白/纯分隔符/多语言/emoji-ZWJ/组合符/随机 UTF-8；`sentence_breaks` 三态各跑一次。
2. **句界三断言**：段界集合 ⊇ 句界；无段落跨越句界；与 `Sequence` 参照逐例相等（含 offsets）。必须含 §1 的反例 `"a\r\nb\n\nc"`。
3. **标准一致性**：`make data/SentenceBreakTest.txt` 逐条比对 `÷`/`×`；本方案可写"恰好相等"（run 列表融合方案只能写"⊇"）。
4. **前置实测**：`split_sentence_bound_indices` 对 `"a\r\nb\n\nc"`、`"段一。\n\n段二。"` 的输出，统一与 `sentence.rs` 的口径。
5. **规则矩阵 + 双引擎**：§2 表全部用例 + 多语言 golden（中/英/日/韩/藏/印地/法 NBSP/阿拉伯/CRLF/`\u{2028}`）；`onig` 下**逐脚本逐分支行为鉴别**取 golden（"编译通过 ≠ 正确"），`unicodedata` 码点卫生断言防同形异码，fancy-regex 用录制基线 + diff。
6. **性能**：release、含 FFI、1KB/100KB/1MB 中位数，三方对比 `sentence_breaks=false` / `=true` / 两级 `Sequence`。

## 6. 风险

1. **SB6/SB7/SB5 无法保证**（两级方案同样不能，唯一语义缺口）：`3.14` 不吸收 `.` 会被切三片（SB6，由 `X_NUM` 覆盖）；`U.S.A.` 与 `left.Now` 同形，上下文无关的正则只能二选一（SB7，作为已声明偏差）；跨脚本吸收失败 `"a\u{0301}你"`（SB5，可用切点吸附到 `grapheme_indices` 加固）。
2. **`Split` 语义收窄**：`pattern` 作用域变成句内；靠文档 + 等价性断言兜住。
3. **口径未实测**：`sentence.rs` 与旧文档对换行归属矛盾，先实测再写断言。
4. **脚本白名单**（偏差 ④）：可闭合但需维护（离线生成 + `sync_check.py`，退出码 0/1/2）。
5. **`\p{...}` 零先例 + `\p{N}` 口径**：属性表版本由引擎决定，可能与 `scripts.rs` 不一致（靠逐脚本鉴别 + golden 锁定）；`\p{N}` = `Nd|Nl|No`，与 `char::is_numeric()` 一致（`½`/`²`/`Ⅻ` 也算数字，同 `digits.rs`），要严格十进制需换 `\p{Nd}` 并声明偏离。
6. 泰文/高棉文隐式句界不在范围；`CLASS_REGEX` 与 `Digits`/`UnicodeScripts`/`Punctuation` 语义重叠，文档须说明本配置是这三者的无损替代，避免叠加互相抵消。

## 7. 性能优化

基线（实测，136 分支大正则 + 单节点，含 FFI）：6.2 / 5.2 / 4.0 MiB/s；编译约 2ms。**分支数不是主要矛盾**（136 分支只慢约 20%，不值得退回白名单），**趟数是**（多节点 `Sequence` 仅单节点约 47%），且吞吐随输入变大而下降 ⇒ 瓶颈在分配与缓存局部性，不在 FFI。

| 档 | 动作 | 预估 |
|---|---|---|
| 1 | 正则线性化 `S (?:JUNK*S)*` → `(X TRAIL*)+`（`JUNK*` 是回溯源）；交替分支按频率排序 | 12~35% |
| 1 | 廉价预筛（句内无标点/数字/异脚本则整体成段）；无 SATerm 时跳过句界扫描 | 10~40% |
| 2 | 去掉句子级中间对象（按全局 offsets 一次 slice）；按句界分块处理长输入 | 10~30% |
| 3 | 下游：保持 `MergedWithNext`（split 数最少）、BPE `cache`、`encode_batch`、truncation early-exit | 可达数倍 |

**禁止**：拆空白分支、用 `(?>...)`/占有量词（加剧双引擎差异）、为性能改写字形。**守卫**：先 profile 定位（`find_matches` / fold+slice / Model / FFI），1、2 档每步都过 §5 的断言与 golden。

## 8. 实施步骤

0. 实测分句口径（§5.4），统一文档与 `sentence.rs`。
1. `split.rs` 改动（§3）+ `sentence.rs` 补 `dedup`。
2. `cargo test --lib split`：§5.1 + §5.2（含反例）。
3. `make data/SentenceBreakTest.txt` + 一致性测试；§5.5 矩阵 + 逐脚本鉴别 + 双引擎基线。
4. Python binding（§4）→ `maturin develop` → `pytest -k split` → `make check-style` 查 stub diff；Node（可选）。
5. 性能三方基线，按 §7 顺序优化。
6. 更新两套文档源（`docs/source/` 与 `docs/source-doc-builder/`，不会互相同步）：脚本白名单、§2 的偏差、风险 1 的缺口、`pattern` 作用域收窄。
7. （可选）注册 `SentenceSplit`：`mod.rs` 7 处 + Python 4 处 + 测试。

---

关键文件：`pre_tokenizers/split.rs`（唯一改动）、`pre_tokenizers/sentence.rs`（已实现未注册）、`tokenizer/pre_tokenizer.rs:73-103`（递归边界）、`tokenizer/normalizer.rs:694-783`、`:745-766`（fold / `MergedWithNext`）、`tokenizer/pattern.rs:63-83`（run 契约）、`bindings/python/src/pre_tokenizers.rs:416-484`（`PySplit` 参照）、`Cargo.toml:84`（`unicode-segmentation` 已在依赖树）。
