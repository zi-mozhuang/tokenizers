# 预分词方案性能测试（已验证定案）

本文档记录单 `Split` 预分词方案的性能实测数据、验证状态与维护方式，以及在其之上的模型层（BPE / WordPiece / Unigram）吞吐与词表特性，作为性能测试的专用文件。方案本体与规则定义见 `tokenizer-architecture.md` 的「预分词方案（已验证定案）」一节；模型算法内部机制见其「§5 Model 四选一」（BPE 合并堆 / WordPiece 贪心最长匹配 / Unigram Viterbi·trie DP）。

## 方案配置

单 `Split` 配置实现预分词四条规则（tokenizers 0.23.2 预编译轮实测，仓库零改动）：

```python
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split
pretok = Split(Regex(TOKEN_RE), behavior="removed", invert=True)
```

`TOKEN_RE = WS*CORE | WS+ | [\s\S]`，`CORE` 为各分支 `(X TRAIL*)+`（分支内可跨 mark 延续、分支间不互串）：`[\p{P}\p{S}]`（标点整段）、`\p{N}`（数字整段）、`[\p{Han}\p{Hiragana}\p{Katakana}\u30FC]`（CJK 组，沿用 `UnicodeScripts::fixed_script` 假名并入语义）、135 脚本全枚举（由 `scripts.rs` 离线生成）、兜底 `\p{L}`；`TRAIL=(?:\p{M}|\u200D)*`（结合符/ZWJ 跟前片）；`WS=(?:\s|\u3000)`；末分支 `[\s\S]` 兜底 guarantee tiling（出现即独立成片，永不静默丢字符）。完整正则见 `/tmp/opencode/pretok_cmp/token_v2.json`（4818 字符，136 分支，一次编译约 2ms）。

## 行为规则映射（均为实测）

- 标点独立连续合并：`ab!!cd→[ab,!!,cd]`，`你好，世界！！→[你好，，世界，！！]`
- 语言分离：`Hello你好→[Hello,你好]`；真泰老 `U+0E01/U+0E81`、真亚美尼亚/格鲁吉亚 `U+0570/U+10D2` 码点已验分开
- 数字分离内部连续：`abc123你好456`，20 位长数字整片留给 Model，`1️⃣` 整体
- 空格贴后不替换：`a  b→[a,  b]`，`hi →[hi, ]` 尾空格独立，`"   "` 整体保留，跨语言 `Hello 你好→[Hello, 你好]`

共享字符行为（确定性三规则：同脚本延续、Common 恒断裂、TRAIL 向前）：

- 同脚本共享融合（中日汉字、拉丁各语言）
- 同形异码按码点切（拉丁 A/西里尔 А）
- Common 作防火墙：`ABC123あいう→[ABC,123,あいう]`，`3.14→[3,.,14]` 按规则字面拆分

## 性能数据

release 轮（含 FFI）：

- 吞吐量中位数约 **6.2 / 5.2 / 4.0 MiB/s**（对应输入规模 1KB / 100KB / 1MB）
- 约为小正则版的 **80%**
- 约为 `Sequence` 版的 **1.7×**
- 正确性门 **34 用例零失败** + **全语料 tiling**

## 关键否定结论（性能与正确性权衡）

- `Sequence` 拼装不可用——`UnicodeScripts` 吞纯空白片（`pre_tokenizer.rs` 单点 `windows(2)` 为空）致丢数据，且跨片无法合并附着；单 `Split` 以 token 正则一次成形，无此问题。
- `onig` 的 `\p{脚本名}` 须逐个行为鉴别（编译通过≠正确）；harness 对测试串加 `unicodedata` 码点卫生断言（曾抓出泰老/亚美尼亚同形字污染）。

## 验证状态

（2026-09，tokenizers 0.23.2 / onig 引擎 / token_v2.json 已固化）

- 正确性门 34 用例零失败 + 全语料 tiling
- 同步检查 `sync_check.py` 通过（135 脚本无漂移）

## 后续维护

- `tokenizers` 升级或 Unicode 数据更新后跑 `python3 sync_check.py`（退出码 0 同步 / 1 有新增 / 2 有删除）
- 新增脚本出现则重跑 `v2.py` 重生成 + `v2eval.py` 全门禁，通过才更新 `token_v2.json`
- wasm/fancy-regex 引擎仅在有 wasm 路线图时验证（服务端默认 onig 已覆盖），届时用 `/tmp/opencode/pretok_fancy/` 方案（fancy 单构建 + 录制基线 diff）执行

## harness 全套

`/tmp/opencode/pretok_cmp/`（`run.py/v2.py/v2eval.py/sync_check.py/probe*.py/bench_models.py/bench_pretok_opts.py/vocab_stats.py` + `corpus.json/results*.json/results_models.json/results_models_bytefallback.json/results_pretok_opts.json/results_vocab_stats.json/token_v2.json`）。运行环境：该目录内置 `.venv`（Python 3.13 + `tokenizers 0.23.2-dev.0` 源码可编辑安装，含 `split.rs` 的 `Arc<SysRegex>` 补丁，由 `maturin develop --release` 重建；原 `0.23.2` 预编译轮无此补丁），含 `models`/`trainers`，无需另建虚拟环境——直接用 `.venv/bin/python` 跑脚本即可。

> **环境变更（2026-09-21）**：`/tmp/opencode/pretok_cmp` 在 OOM 重启后被清（含 `.venv`）。后续工作迁至仓库持久目录 `dataset/pretok_cmp/`，改用 `sys.path.insert(0, bindings/python/py_src)` 直接导入 `tokenizers`（不再依赖 `.venv`）；横评脚本见 `bench_hf_official.py` / `bench_hf_pretok.py`（见上「横向对比」节）。

## 模型层性能（BPE / WordPiece / Unigram）

在同一预分词环境（单 `Split` + `token_re_v2`、`normalizer=None` 无损管线）下，对各分词模型训练至目标词表 30k（`bench_models.py`），再按 `run.py` 同一口径（1KB:(50,5) / 100KB:(5,5) / 1MB:(2,3)，取最优、MiB/s 中位数，7 类语料 en/zh/mixed/digit_punct/ws_dense/adversarial/salad）测 `encode` 吞吐。模型算法内部机制见 `tokenizer-architecture.md` 的「§5 Model 四选一」。

### 30k 真实词表（自然语料 train_corpus.txt，byte_fallback 开启）

训练于 en+zh+mixed 自然语料（约 34 MB，字符级去重上限 50M），词表真正逼近 30k（`results_models_bytefallback.json`）：

| 模型 | 1KB | 100KB | 1MB | 训练后词表 |
|---|---|---|---|---|
| BPE | 5.43 | 4.68 | 3.49 | 30256 |
| WordPiece | 5.65 | 4.22 | 3.77 | 30000 |
| Unigram | 3.71 | 3.04 | 3.23 | 30256 |

### 词表特性比较（vocab_stats.py，同配方训练到 30k）

| 模型 | 词表 | 单字 token | 字节回退 token | 平均长度 | 压缩(chr/tok) | 压缩(byte/tok) | EN chr/tok | ZH chr/tok |
|---|---|---|---|---|---|---|---|---|
| BPE | 30256 | 10158 | 256 | 3.46 | 2.95 | 4.32 | 4.20 | 1.64 |
| WordPiece | 30000 | 10158 | 0 | 3.09 | 2.66 | 3.89 | 3.83 | 1.47 |
| Unigram | 30256 | 10158 | 256 | 3.65 | 2.60 | 3.80 | 3.37 | 1.59 |

token 长度分布（按字符数）：BPE `1:10158 2:5726 3:2757 4:2664 5+:8951`；WordPiece `1:10158 2:693 3:10419 4:3107 5+:5623`；Unigram `1:10158 2:5407 3:2650 4:2218 5+:9823`。

- 三者共享 **10158 个单字 token**（同一字符表），差异全在子词合并策略。
- 压缩率 **BPE > WordPiece > Unigram**（chr/tok 2.95 > 2.66 > 2.60）；BPE 既压得最紧、吞吐又与 WordPiece 并列最快；Unigram 最慢且压得最松。
- 中英差异是真实成本驱动：英文 chr/tok ≈ 3.4–4.2（词级），中文仅 ≈ 1.5–1.6（一字一 token，CJK 无空格），同等字符数下中文约消耗 **2.5×** 的 token 数；三者里 BPE 对中文压缩最好（1.64），WordPiece 最差（1.47）。
- byte_fallback 在本语料不触发：UNK% = 0、字节 token 使用率 = 0 三者皆然——30k 词表已覆盖全部字符，回退仅作未见稀有字符兜底（用 `𝄞𠮷` 已验证生效）。

### 旧合成小语料对照（为何需要更大语料）

早期 `corpus.json` 为重复合成句，词表仅收敛到 BPE 159 / WordPiece 207 / Unigram 76（关闭 byte_fallback，`results_models.json`）：

| 模型 | 1KB | 100KB | 1MB | 训练后词表 |
|---|---|---|---|---|
| BPE | 4.30 | 4.71 | 3.09 | 159 |
| WordPiece | 5.21 | 4.75 | 3.13 | 207 |
| Unigram | 3.79 | 3.05 | 2.30 | 76 |

（开启 byte_fallback 旧数据：BPE 4.76/4.08/2.61 词表 415、WordPiece 4.49/4.44/3.81 词表 207（不支持回退）、Unigram 3.59/3.38/2.50 词表 332。）小词表下差距被严重压缩、且远未到真实大模型 30k–100k 量级；换用自然语料后（上表）差异才如实放大。

### 结论

- 模型阶段非瓶颈：对比预分词器基线 6.2 / 5.2 / 4.0 MiB/s，`encode = 预分词 + 模型`，模型仅增加约 1–3 MiB/s 开销，绝大多数成本在预分词阶段。
- 排序 **WordPiece ≈ BPE > Unigram**：WordPiece 单趟左→右贪心最长匹配（见架构 §5.2）；BPE 需多轮堆合并（§5.1）；Unigram 跑 Viterbi / trie 前缀 DP（§5.4），单 token 成本最高。
- `adversarial` 语料 WordPiece 异常快（15–21 MiB/s）属离群点（重复片段命中缓存），不代表常态。
- 词表偏小是主要偏差（合成语料重复度高），真实大模型（30k–100k）下三者差距会被放大。

### byte_fallback 实现要点（0.23.2 发布轮坑）

- `tokenizers 0.23.2` 发布轮里 byte_fallback **不会自动生效**：`BpeTrainer`/`UnigramTrainer` 既不读模型 `byte_fallback` 标志、也不接受 `initial_alphabet` 注入 256 个字节 token；推理层 byte_fallback 仅当 vocab 已含字节 token 时才触发，且字节 token 格式为**小写 `<0xHH>`**（仓库源码 `#04X` 大写属另一版本）。
- 启用方式：训练后做 `tokenizer.json` 手术——注入 256 个 `<0xHH>` 并把 `model.byte_fallback=true`，再 `from_file` 重建（`bench_models.py` 的 `enable_byte_fallback`）。自检：未见字符 `𝄞𠮷` 被分解为 `<0xF0><0x9D>...`。
- **WordPiece 模型无 byte_fallback 选项**（未知字符仍走 `[UNK]`），属设计限制而非配置问题。
- 开启后主要收益是正确性（未知字符无损转为 UTF-8 字节 token，消除 `[UNK]` 信息丢失）；本语料未知字符极少，吞吐基本持平（BPE 1MB 因字节 token 增多略降）。

## 横向对比：C1/C2 与项目生产方案 + HuggingFace 主流配置（2026-09-21）

为锚定项目方案定位，把 C1/C2（CLASS_REGEX + merged_with_next）、C3（生产 TOKEN_RE + removed + invert）与 **HuggingFace 官方 tokenizer** 放在同一语料上横评「分词效率（吞吐 + 粒度）+ 压缩率」。

### 对比对象与口径
- **C1** = `Split(CLASS_REGEX, merged_with_next, sentence_breaks=True)`；**C2** = 同式 `sentence_breaks=False`；**C3** = `Split(TOKEN_RE, removed, invert=True)`（生产版）。C1/C2/C3 均用**全量 270MB 语料流式训练**至 30k BPE（release 构建）。
- **HF 官方**：直接下载/复用官方 `tokenizer.json`（**不再自己实现 + 从头训**），含 BERT(bert-base-uncased)、GPT-2、RoBERTa、Llama(hf-internal-testing/llama-tokenizer)、T5、XLM-R、Qwen3-0.6B。各 vocab 不同（30k–250k）。
- **吞吐/粒度**：端到端 `encode` 在 1KB/100KB/1MB 三档（取最优 MiB/s）+ 1MB 预分词片段数/千字符。
- **压缩率**：在**分层 8,153,035 字符样本**（跨全文每 k 行取一条，比例稳定 <0.1%）上 encode，统计 chars/token。

### 结果
| tokenizer | 家族 | vocab | 1KB | 100KB | 1MB | 片段/1k字符 | chars/token |
|---|---|---|---|---|---|---|---|
| **C2_CLA_nosb** | 我们(无句界) | 30000 | **11.11** | **6.48** | 4.71 | 133.0 | **3.199** |
| **C1_CLA_sb** | 我们(句界) | 30000 | 10.64 | 6.56 | **5.11** | 136.6 | **3.206** |
| C3_prod_token_re_v2 | 我们(生产) | 30000 | 5.57 | 4.22 | 3.36 | 210.7 | 2.999 |
| HF_t5-base | T5(SentencePiece) | 32100 | 3.04 | 2.51 | 2.81 | 124.6 | **3.744** |
| HF_gpt2 | GPT-2(ByteLevel) | 50257 | 2.98 | 2.33 | 2.01 | 212.5 | 1.574 |
| HF_roberta-base | RoBERTa(ByteLevel) | 50265 | 2.17 | 2.23 | 1.61 | 212.5 | 1.574 |
| HF_llama-tokenizer | Llama(ByteLevel) | 32000 | 5.04 | 3.36 | 2.66 | *733.0¹* | 1.847 |
| HF_xlm-roberta-base | XLM-R(SentencePiece) | 250002 | 2.77 | 2.14 | 1.81 | 124.6 | 2.641 |
| HF_bert-base-uncased | BERT(WordPiece) | 30522 | 1.95 | 2.32 | 1.76 | 199.0 | 2.626 |
| HF_Qwen3-0.6B | Qwen3(ByteLevel) | 151669 | 2.14 | 2.10 | 1.77 | 218.4 | 2.658 |

¹ Llama 官方 `tokenizer.json` 的 `pre_tokenizer=null`（SentencePiece 转换版把切分放进 normalizer），其"片段/1k字符"只能按 token 计数，不可与"预分词片段"直接横比。

### 结论
- **分词效率（吞吐）**：C1/C2 全场最快，比最快的 HF 还快约 2×（1KB 10.6–11.1 vs GPT-2 2.98）；1MB 档 C1 5.11 仍居首。C3（生产大正则）仅约 C1/C2 的一半（5.57/4.22/3.36）——大正则 + 最细粒度双重拖累。注：C1/C2 的 `normalizer=None`（无归一化趟），而 BERT/T5/XLM-R 带 normalizer，部分成本记在它们账上。
- **粒度**：最粗 T5/XLM-R(124.6, Metaspace) > C2/C1(133/136.6)；最细 C3(210.7)、GPT-2/RoBERTa(212.5)、Qwen3(218.4)。
- **压缩率**：冠军 T5(SentencePiece, 3.744)，C1/C2 以 3.20 紧随，且**压过所有 byte-level 系**（GPT-2/RoBERTa 1.574、Llama 1.847、Qwen3 2.658）、BERT(2.626) 与 XLM-R(2.641)；并**领先自家生产版 C3 约 7%**（3.20 vs 3.00）。
- **byte-level 系垫底主因**：语料近半中文，字节级把 CJK 炸成字节 token（GPT-2/Llama 仅 1.57–1.85 字符/token）。
- **T5 登顶 = 数据/词表优势**：其 SentencePiece 词表在海量多语料上训练，中文覆盖远好于我们 270MB-only 的 30k，不纯是算法。
- **句界中性**：C1 vs C2 压缩仅差 0.2%（3.206 vs 3.199），吞吐在 release 下互有 ~4% 浮动（debug 下 UAX29 扫描相对开销被放大到 ~15%）——与 `uax29-sentence-breaks-split.md` §10 评估一致。

### 关键经验（环境 + 方法）
- **用真实官方 tokenizer 当基线**：HF 家族直接 `Tokenizer.from_file(官方tokenizer.json)`，无需自己实现 + 从头训 BPE，结果更忠实。环境已有 `Qwen3-0.6B/tokenizer.json`，其余 curl 下载至 `dataset/pretok_cmp/hf_tok/`。
- **release 构建**：`cargo build --release` 增量 27s 完成，已装回 `py_src/tokenizers/tokenizers.abi3.so`（原 149MB debug 版备份为 `tokenizers.abi3.so.debug.bak`）；吞吐口径才与本文 §性能数据 一致（debug 约慢 4×）。
- **内存安全**：BPE 训练**流式** `train_from_iterator(逐行)`，绝不整份语料常驻（曾因 `f.read()` 全量常驻 + trainer 词频表触发 OOM，并连带清空 `/tmp` 工作区）。每个配置**独立进程**训练/测量，退出即释放内存（单程峰值 ~2.5GB）。
- **压缩率测在分层采样上**：比例稳定，比逐字节 encode 全量快 ~10×；全量 estimate 用线性外推即可。
- **工作区落地**：`/tmp/opencode/pretok_cmp` 在 OOM 重启后被清，脚本/词表已迁至仓库持久目录 `dataset/pretok_cmp/`，并改用 `sys.path.insert(0, bindings/python/py_src)` 直接导入（不再依赖 `.venv`）。脚本：`bench_hf_official.py`、`bench_hf_pretok.py`；产物：`tokenizer_C{1,2,3}_*.json`、`results_hf_official.json`。

### 与 §性能数据 的衔接
本文 §性能数据 的 6.2/5.2/4.0 MiB/s 即 C3（生产大正则单 Split）口径；本横评 C3 实测 5.57/4.22/3.36（同 release 构建），量级一致。C1/C2 较 C3 约快 **1.9×**，印证"简化正则 + 较粗粒度"在吞吐上的收益。

## sentence_breaks 扩展（2026-09-21，复用共享环境）

在共享环境 `/tmp/opencode/pretok_cmp/` 内实现并验证 `Split.sentence_breaks`（UAX #29 句界预分词，见 `uax29-sentence-pretokenizer.md`）。复用既有 `.venv` 与可编辑安装，未另建环境。

### 对测试环境所做的修改（需记录）
1. **安装 pytest**：`/tmp/opencode/pretok_cmp/.venv/bin/pip3 install pytest`（原环境无 pytest，无法跑 `tests/`）。
2. **重建扩展（debug）**：在 `bindings/python/` 下执行
   `VIRTUAL_ENV=/tmp/opencode/pretok_cmp/.venv /tmp/opencode/pretok_cmp/.venv/bin/maturin develop`
   此前该环境为 `maturin develop --release` 构建；现改为 debug 构建（含 sentence_breaks 特性）。如需恢复 release 性能基准，重跑 `maturin develop --release` 即可。

   （2026-09-21 横评后已重跑 `cargo build --release` 把 release `.so` 装回 `bindings/python/py_src/tokenizers/tokenizers.abi3.so`，149MB debug 版备份为 `tokenizers.abi3.so.debug.bak`；性能基准口径以 release 为准，见上「横向对比」节。）
3. **`.pyi` 手动同步**：`make check-style` 的 stub-gen（`tools/stub-gen`）需联网拉取 crates，本环境离线无法运行；已手动在 `py_src/tokenizers/pre_tokenizers.pyi` 的 `Split` 增加 `sentence_breaks` 参数与属性 getter/setter。合入主分支后须用 `make check-style` 重新生成以消差（勿手改残留）。
4. **新增 Python 测试**：`tests/bindings/test_pre_tokenizers.py::TestSplit::test_sentence_breaks`（沿用既有文件，未新建文件）。Rust 测试内联于 `tokenizers/src/pre_tokenizers/split.rs` 的 `mod tests`（新增 5 用例，连同既有共 13 passed）。

### 与文档所述 `Arc<SysRegex>` 补丁的差异
本文档 §harness 称该环境含 `split.rs` 的 `Arc<SysRegex>` 补丁，但本仓库当前 `split.rs` 为 plain `SysRegex`，已装 `.so` 的 `Split` 亦为 3 参 `(pattern, behavior, invert=False)`——即该补丁未落入本 checkout。`sentence_breaks` 实现嫁接于 plain `SysRegex`，与现有 `.so` 一致、无冲突。若后续合入 `Arc<SysRegex>` 补丁，本改动仅需把字段旁 `regex: SysRegex` 一并改为 `Arc<SysRegex>`。

### 验证命令（均用共享 venv）
- Rust：`cd tokenizers && cargo test --lib split`（13 passed；`cargo fmt --check` / `cargo clippy --lib -D warnings` 通过）
- Python：`/tmp/opencode/pretok_cmp/.venv/bin/python -m pytest bindings/python/tests/bindings/test_pre_tokenizers.py -k Split`（5 passed）
- 示例脚本（目标管线组装 + 落盘断言 + 往返无损）：`/tmp/opencode/pretok_cmp/.venv/bin/python bindings/python/examples/train_uax29_bpe.py`
  - 输出确认：`"normalizer": null`、`"post_processor": null`、`model.byte_fallback: true`、`pre_tokenizer.sentence_breaks: true`，且 7 句语料 encode→decode 无损。

### 行为要点（实测）
- `sentence_breaks=True` 时先按 UAX #29 句界切片，再在句内跑 `CLASS_REGEX` + `behavior`，句界用 `slice` 构造，零字符增删改，故句界永不被 `pattern` 跨越；`invert`/`behavior` 均作用于句内。
- `byte_fallback` 为**推理期**属性：`BpeTrainer` 不接受该选项、其 `initial_alphabet` 只取每项首字符，无法注入 `<0xXX>`；训练侧仅由 `initial_alphabet`（来自语料字符集）保证覆盖。示例脚本据此用 `unk_token="<unk>"` 兜底并注明。
