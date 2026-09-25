# 预分词性能（最终 C2）

**定案**：最终最小路径是 C2 `CLASS_REGEX` + `Split(..., behavior="merged_with_next", invert=false)`。C3 `TOKEN_RE` 大正则只作参照；C1 `sentence_breaks` 是历史实验，当前 Rust `Split` 无此字段。规则、最终正则和行为见 `uax29-sentence-pretokenizer.md`。

## 1. C3 参照与基线

C3 配置如下；`TOKEN_RE` 的完整定义在 `dataset/pretok_cmp/token_v2.json`：

```python
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split
pretok = Split(Regex(TOKEN_RE), behavior="removed", invert=True)
```

`TOKEN_RE = WS*CORE | WS+ | [\s\S]`。`CORE` 覆盖标点、数字、CJK、135 个脚本及字母兜底；各类使用 `(X TRAIL*)+`，可吸收组合符/ZWJ，末分支保证 tiling。C3 结果只用于对照。

历史单 Split/C3 基线（release、含 FFI；独立轮次）：

| 输入规模 | 1KB | 100KB | 1MB |
|---|---:|---:|---:|
| 吞吐（MiB/s） | 6.2 | 5.2 | 4.0 |

## 2. 既有模型层基准

既有轮次使用约 34MB 自然语料，字符去重上限 50M；`train_from_iterator` 流式训练，`normalizer=None`，按三档输入和 7 类语料测 `encode`，取最优 MiB/s 中位数。

### 吞吐

| 模型 | 1KB | 100KB | 1MB |
|---|---:|---:|---:|
| BPE | 5.43 | 4.68 | 3.49 |
| WordPiece | 5.65 | 4.22 | 3.77 |
| Unigram | 3.71 | 3.04 | 3.23 |

### 词表特性

| 模型 | 词表 | 单字 token | 字节回退 token | 平均长度 | chr/tok | byte/tok | EN chr/tok | ZH chr/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BPE | 30256 | 10158 | 256 | 3.46 | 2.95 | 4.32 | 4.20 | 1.64 |
| WordPiece | 30000 | 10158 | 0 | 3.09 | 2.66 | 3.89 | 3.83 | 1.47 |
| Unigram | 30256 | 10158 | 256 | 3.65 | 2.60 | 3.80 | 3.37 | 1.59 |

WordPiece 与 BPE 吞吐接近，Unigram 最慢；BPE 压缩最好。中文 token 数约为英文的 2.5 倍。该语料字符均被词表覆盖，`byte_fallback` 使用率为零；未见字符测试仍能触发回退。`adversarial` 的异常高吞吐是缓存离群点，不代表常态。

## 3. `byte_fallback` 限制

- `tokenizers 0.23.2` 中，`BpeTrainer`/`UnigramTrainer` 不接受 `byte_fallback` 训练选项，也不会自动注入字节 token；`initial_alphabet` 只接收单字符。
- 推理期只有词表含完整字节 token 时才触发；格式为小写 `x`、大写 hex 的 `<0xHH>`。需要任意字节覆盖时，训练后注入 256 项、设置 `model.byte_fallback=true`，再 `from_file` 重建；用 `𝄞𠮷` 做未知字符自检。
- WordPiece 没有 `byte_fallback` 选项，未知字符仍走 `[UNK]`。该能力主要用于未知字符无损保留；未知字符很少时吞吐基本持平。

## 4. C1/C2/C3 与 HuggingFace 横评

C1 是历史句界实验，C2 是同一 `CLASS_REGEX` 的无句界方案，C3 是 `TOKEN_RE` 大正则方案。当前源码的 `Split` 不接受 `sentence_breaks`，所以 C1 只用于解释历史产物，不能按当前 API 重建。下表来自 `dataset/pretok_cmp/results_hf_official.json` / `dataset/pretok_cmp/bench_official.log`：C1/C2/C3 用约 270MB 语料流式训练 30k BPE，HF 使用官方 `tokenizer.json`；压缩率取 8,153,035 字符分层样本。

| tokenizer | 家族 | vocab | 1KB | 100KB | 1MB | 片段/1k字符 | chars/token |
|---|---|---:|---:|---:|---:|---:|---:|
| C2_CLA_nosb | 我们（无句界） | 30000 | 11.11 | 6.48 | 4.71 | 133.0 | 3.199 |
| C1_CLA_sb | 我们（历史句界） | 30000 | 10.64 | 6.56 | 5.11 | 136.6 | 3.206 |
| C3_prod_token_re_v2 | 我们（C3 参照） | 30000 | 5.57 | 4.22 | 3.36 | 210.7 | 2.999 |
| HF_t5-base | T5（SentencePiece） | 32100 | 3.04 | 2.51 | 2.81 | 124.6 | 3.744 |
| HF_gpt2 | GPT-2（ByteLevel） | 50257 | 2.98 | 2.33 | 2.01 | 212.5 | 1.574 |
| HF_roberta-base | RoBERTa（ByteLevel） | 50265 | 2.17 | 2.23 | 1.61 | 212.5 | 1.574 |
| HF_llama-tokenizer | Llama（ByteLevel） | 32000 | 5.04 | 3.36 | 2.66 | 733.0¹ | 1.847 |
| HF_xlm-roberta-base | XLM-R（SentencePiece） | 250002 | 2.77 | 2.14 | 1.81 | 124.6 | 2.641 |
| HF_bert-base-uncased | BERT（WordPiece） | 30522 | 1.95 | 2.32 | 1.76 | 199.0 | 2.626 |
| HF_Qwen3-0.6B | Qwen3（ByteLevel） | 151669 | 2.14 | 2.10 | 1.77 | 218.4 | 2.658 |

¹ Llama 官方 `tokenizer.json` 的 `pre_tokenizer=null`，片段数实际按 token 统计，不能直接与其他预分词片段数横比。

C2/C1 吞吐领先，C3 受大正则和细粒度影响约为其一半；T5 压缩率最高，C1/C2 次之且优于 C3。句界压缩近中性，但 release 小输入约慢 4%，故最终采用 C2。C1/C2 无归一化，部分 HF 配置带 normalizer，口径不能完全等同。

JUNK 拖尾的对照表与决策见 `uax29-sentence-pretokenizer.md`；结果文件为 `dataset/pretok_cmp/results_cmp_junktrail.json` 和 `dataset/pretok_cmp/results_cmp_junktrail_effic.json`。

## 5. 复测入口

现存入口均在仓库内：

- 官方 tokenizer 横评：`dataset/pretok_cmp/bench_hf_official.py`；读取现存 `tokenizer_*.json`，脚本含本机绝对路径，需校准后运行。
- JUNK 对照：`dataset/pretok_cmp/cmp_junktrail.py`；`effic` / `report` 模式。它直接加载固定路径的 `tokenizers.abi3.so`，依赖现存旧编译 binding，不是当前 Rust API 的证明。
- PyPI 独立验证：`design/minimal-tokenizer-path.md` 的 `tokenizers==0.23.2` 脚本；在仓库外替换语料路径后运行，预分词正则以 `uax29-sentence-pretokenizer.md` 为准。
- 当前源码回归：在 `tokenizers/` 运行 `cargo test -p tk-encode split`；这是通用 `Split` 测试，不是 `CLASS_REGEX` 专用测试。

`bench_hf_pretok.py`、`train_and_compress.py` 仍调用 `Split(..., sentence_breaks=...)`，`run_full.sh` 间接调用前者；当前 `Split` 没有该参数，三者只能作历史脚本/结果索引，不能原样作为当前复测入口。吞吐基准使用 release 构建；升级 `tokenizers`、Unicode 数据或正则引擎后重跑门禁。
