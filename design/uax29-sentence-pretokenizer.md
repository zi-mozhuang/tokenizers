# 字符类 + Split 预分词方案（最终版，无句界）

> 本文档为最终方案，替代有句界版 `design/uax29-sentence-breaks-split.md`。
> 最终配置 **不使用句界层**：`Split(Regex(CLASS_REGEX), behavior="merged_with_next")`。
> 句界扩展 `sentence_breaks` 已实现并验证（见 §10），但经横向评测判定为「压缩中性 + 吞吐负收益」，故默认关闭、不纳入最终管线。

---

## 0. 一句话结论

预分词 = 单个 `Split` + 字符类正则 `CLASS_REGEX` + `behavior="merged_with_next"`：按「汉字组 / 拉丁 / 数字」三大类切分，类内允许拖尾非类字符延续（标点、组合符、ZWJ 等），类间不互串。不引入句界层，故也无需 UAX #29。

- 改动面：仅 `CLASS_REGEX` 的定义与 `Split` 的常规配置；不新增类型、模块或序列化字段。
- 不动：`sentence.rs` / `mod.rs` / Node binding；序列化沿用 `Split` 既有 `to_str()/from_file`。

---

## 1. 目标与动机

1. **字符类分离**：中日韩 / 拉丁 / 数字各成 run，避免中英文或数字混入同一 token。
2. **复用现有 `Split`**：plain `SysRegex`，最小改动、向后兼容、天然可序列化。
3. **可序列化**：产出的 `tokenizer.json` 必须能被 `Tokenizer.from_file` 还原（硬约束）。

---

## 2. pattern（CLASS_REGEX）

文档 §2 给出的字符类正则，作为 `behavior=MergedWithNext`、`invert=false` 的识别器：

```
\p{White_Space}*
(?:
  [\p{Han}\p{Hiragana}\p{Katakana}\u30FC]+
    (?:[^\p{White_Space}\p{L}\p{N}]*[\p{Han}\p{Hiragana}\p{Katakana}\u30FC]+)*
| [\p{Latin}]+
    (?:[^\p{White_Space}\p{L}\p{N}]*[\p{Latin}]+)*
| \p{N}+
    (?:[^\p{White_Space}\p{L}]*\p{N}+)*
)
```

**两个必须锁进断言的细节**（测试覆盖，违反即行为错误）：

1. **JUNK 排除 `\p{N}`**：否则 `abc123def` 会被 JUNK（`[^…\p{N}]*` 不含数字）切断后的数字再接回，塌缩成单 match。
2. **不得用 `[\p{L}]` 兜底**：否则汉字会被吞进 Latin run（中英文不分离）。

`\p{White_Space}` / `\p{Han}` / `\p{Latin}` / `\p{Hiragana}` / `\p{Katakana}` 等属性名在仓库 vendored 的 **oniguruma**（默认 feature `onig`）属性表中已确认存在，无需降级为显式空白类或脚本枚举。

---

## 3. 行为规则映射（均为实测）

- 语言分离：`Hello你好→[Hello,你好]`；真泰老 `U+0E01/U+0E81`、亚美尼亚/格鲁吉亚 `U+0570/U+10D2` 码点已验分开。
- 数字连续内部：`abc123你好456`，长数字整片留给 Model，`1️⃣` 整体。
- 空格贴后不替换：`a  b→[a,  b]`、`hi →[hi, ]` 尾空格独立、跨语言 `Hello 你好→[Hello, 你好]`。
- 标点：本方案按三大类粗切，标点落在 JUNK 拖尾内（与生产版 `token_v2.json` 的脚本/标点分支语义一致，但本方案不枚举全部脚本）。

共享字符行为（确定性三规则：同脚本延续、Common 恒断裂、TRAIL 向前）：同脚本共享融合（中日汉字、拉丁各语言）；同形异码按码点切（拉丁 A/西里尔 А）；Common 作防火墙 `ABC123あいう→[ABC,123,あいう]`、`3.14→[3,.,14]` 按规则字面拆分。

---

## 4. 序列化与兼容

```rust
pub fn new(pattern, behavior, invert: bool) -> Split  // 默认 invert=false
```

向后兼容：旧 `tokenizer.json` 反序列化无 `sentence_breaks` 字段时默认 `false`（即使存在该字段，本方案也不启用）。`Clone` / `PartialEq` 维持原样。

---

## 5. Python 绑定

```python
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split
pretok = Split(Regex(CLASS_REGEX), behavior="merged_with_next")
```

> **注意**：`.pyi` 类型桩由 `make check-style`（`tools/stub-gen`）自动生成。在**离线**环境无法联网拉取 stub-gen 依赖时，需手动同步 `py_src/tokenizers/pre_tokenizers.pyi` 的 `Split` 签名；合入主分支前应跑一次 `make check-style` 覆盖手改。

---

## 6. 目标管线示例（示例脚本）

`bindings/python/examples/train_uax29_bpe.py` 组装如下目标管线并断言落盘与往返无损：

| 组件 | 取值 | 说明 |
|---|---|---|
| `normalizer` | `None` | 原文直通，大小写/重音/全角/空白/控制字符全保留 |
| `pre_tokenizer` | `Split(CLASS_REGEX, merged_with_next)` | 字符类切分（无句界） |
| `model` | `BPE(byte_fallback=True, unk_token="<unk>")` | 字节回退兜底 |
| `post_processor` | `None` | 留给调用方第二步的 bos/eos/template |
| `decoder` | `Sequence([ByteFallback(), Fuse()])` | 推理期还原 |

### 关键经验：`byte_fallback` 是**推理期**属性

- `BpeTrainer` **不接受** `byte_fallback` 选项，其 `initial_alphabet` 只取每项**首字符**，无法注入 256 个 `<0xXX>` 字节 token。
- 因此训练侧覆盖率**只能**靠 `initial_alphabet = sorted(语料字符集)` 保证（示例脚本即如此）；`byte_fallback=True` 仅作推理期安全网。
- 若需真正覆盖任意字节，应在训练后做 `tokenizer.json` 手术：注入 `<0x00>`..`<0xFF>` 并把 `model.byte_fallback=true`，再 `from_file` 重建。
- 统一用 `unk_token="<unk>"` 兜底，避免 `unk_token=None` 且字符既非词汇又无 `<0xXX>` 时被静默丢弃。

脚本断言：`save()` 后读回 JSON 中 `"normalizer" is None`、`"post_processor" is None`、`model["byte_fallback"] is True`；且对训练语料做 `encode→decode` 还原无损。

---

## 7. 测试策略（内联，不建新文件）

全部内联进 `split.rs` 现有 `#[cfg(test)] mod tests`，不新增类型/模块/fixture：

1. `class_regex_basic`：语言分离 + 数字连续 + 空格贴后（§3 用例矩阵）。
2. `class_regex_lossless`：splits 拼接 == 输入、offsets 连续单调、无空 split（空串/纯空白/多语言/emoji-ZWJ/组合符）。
3. `class_regex_junk_excludes_n`：断言 `abc123def` 不被 JUNK 塌缩、`3.14` 按字面拆。
4. `class_regex_no_latin_fallback`：断言汉字不进 Latin run。

---

## 8. 已知偏差（已声明，未修复）

- **脚本白名单未闭合**：`[\p{Latin}]+` 之后跟 `[^\p{White_Space}\p{L}\p{N}]*`，可能把未知脚本的拖尾字符一并吸入当前 run（例如中日韩之外的罕见脚本）。
- **`U.S.A.` / `left.Now` 与 `3.14` 同形歧义**：无句间缩写规则，句点后是否断句存在歧义，与数字小数点无法区分（无句界层后此项归为「不建模」，由 Model 在子词层吸收）。

---

## 9. 过程经验总结（Lessons Learned）

1. **序列化约束决定方案形态**：任何需 `from_file` 还原的预分词都必须落入可序列化节点；本方案即 plain `Split`，无此负担。
2. **离线阻断 `make check-style` 的 stub-gen**：`tools/stub-gen` 需联网拉 crates，离线时只能手改 `.pyi`，合并前务必本地跑 `make check-style` 覆盖。
3. **onig 的 `\p{}` 属性需逐个鉴别**：编译通过 ≠ 语义正确；本方案的属性名在 vendored oniguruma 中实测存在，但换引擎（fancy-regex/wasm）需重验。
4. **`byte_fallback` 训练期无效**：这是 0.23.x 的发布轮坑，训练侧覆盖率只能靠 `initial_alphabet`，示例脚本据此设计。
5. **`MergedWithNext` 合并方向影响断言**：它把每个 match 与其**前置** gap 合并，尾随 gap 成为独立 piece；测试若直接比对 span 会误判，应改为「无损 + 去空白后核心 token 顺序」或「句界包含性」断言。
6. **内存安全**：BPE 训练必须**流式** `train_from_iterator(逐行)`，整份语料常驻 + trainer 词频表曾触发 OOM 并清空 `/tmp` 工作区；每配置**独立进程**训练/测量，退出即释放内存（单程峰值 ~2.5GB）。
7. **工作区持久化**：脚本/词表落在仓库 `dataset/pretok_cmp/`，并 `sys.path.insert(0, bindings/python/py_src)` 直接导入 `tokenizers`，不依赖 `.venv`。

---

## 10. 句界扩展 sentence_breaks（已实现，未纳入最终配置）

`sentence_breaks` 在 `Split` 上挂可选布尔字段：开启后先按 UAX #29 句界切片，再在每句内独立跑 `CLASS_REGEX + behavior`（句界切点由 `slice` 构造，零字符增删改、不跨界、可序列化）。改动面 `tokenizers/src/pre_tokenizers/split.rs` + `bindings/python/src/pre_tokenizers.rs` + `.pyi`；Rust 13 passed、Python 5 passed。

**评测结论（见 `pretok-performance.md` 横向对比）**：

- **压缩中性**：270MB 真实语料上 C1(带句界) vs C2(无句界) chars/token 仅差 ~0.2%（采样 2.620 vs 2.615；官方横评 3.206 vs 3.199）；词表改变约 3.4%（~1027 项），但压缩贡献极小。
- **吞吐负收益**：句界引入 UAX #29 扫描 + 句内二次 `Split`，release 构建下 C1 较 C2 在 1KB 档慢约 4%（debug 构建下相对开销被放大到约 15%）。
- **决策**：最终配置 `sentence_breaks=False`。该字段保留为可选能力（向后兼容、默认关闭），供需要句界隔离的场景（如跨句不可合并约束）按需开启。

---

## 11. 验证结果

| 验证项 | 命令 | 结果 |
|---|---|---|
| Rust 单元 | `cd tokenizers && cargo test --lib split` | **13 passed**（含 5 个 sentence_breaks 用例） |
| 风格/静态 | `cargo fmt --check` + `cargo clippy --lib -D warnings` | 通过 |
| Python 单元 | `pytest tests/bindings/test_pre_tokenizers.py -k Split` | **5 passed** |
| 示例脚本 | `python bindings/python/examples/train_uax29_bpe.py` | 落盘 `normalizer=null`/`post_processor=null`/`model.byte_fallback=true`，7 句语料 encode→decode 无损 |
| 横评 | `dataset/pretok_cmp/bench_hf_official.py` | C2 吞吐 11.11/6.48/4.71、chars/token 3.199（见 `pretok-performance.md`） |

---

## 12. 后续维护提示

- 升级 `tokenizers` / 换正则引擎时，重跑 §7 的内联测试门禁。
- 若确需句界隔离，开启 `sentence_breaks=True`（§10），并补测跨句不可合并约束。
- 性能基准参考 `pretok-performance.md`；本方案 `encode = 预分词 + 模型`，绝大多数成本在预分词阶段（C2 约 11/6.5/4.7 MiB/s，release 构建）。
