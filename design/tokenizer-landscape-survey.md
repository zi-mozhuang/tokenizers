# 主流分词器景观调查

> 数据快照：2026-09-22。目标：从 Hub Top-20k 元数据确定根与家族，筛选接近
> `added_tokens` 预注册 / `normalizer=None` / `CLASS_REGEX+mwn` / `BPE+byte_fallback` 的 tokenizer。

## 1. 方法口径

1. 用 Models List API 游标翻页取下载量 Top-20k（合计 29.2 亿）；`skip>3000` 会被 API 拒绝。
2. 从 `base_model` tag 与 `cardData.base_model` 追根，最多 10 跳；merge 多父取首父。启发式归并 tokenizer 家族。
3. 个例只取 KB 级 `tokenizer_config.json` 和 `tokenizer.json` 头部 Range，不下载权重/blob；验证 normalizer、pre-tokenizer、model、byte fallback。
4. 历史行为对照用 Python `regex` 镜像当时的 `Split` fold，对比 `CLASS_REGEX`、Qwen2、o200k/GPT-2 pattern。
5. 原始 20k JSONL 和早期分析脚本未归档；C1/C2 结果只保留历史证据。当前 checkout 的 `Split` 仅读取 `pattern`、`behavior`、`invert`：旧 `sentence_breaks` 字段会被忽略，`train_and_compress.py` 的 C1/C2 构造也不能用当前绑定复现；C2 JSON 本身不含该字段。本文仅保留最终统计，不宣称可一键复现。

## 2. 覆盖结论（N=20k）

- 去重得到 **12,171 个根**；按根覆盖 95% 下载仍需 **1,867 个**，根粒度不收敛。
- 聚成 **6,207 个家族**；覆盖 95% 下载需 **430 个文本家族**，占全量下载 **90.5%**；尾部约 390 个单例主要是 TTS 音色、OCR、分类器。
- 经济截断：文本下载量口径 **Top-15=71.1%**，**Top-35=82.1%**。

## 3. 评分、行为与排除线

- A：`added_tokens` 预注册；文本系全有，CTC/纯视觉/音频/蛋白/音乐无。
- B：`normalizer=None`；`✓` 直通，`~` NFC/空 Sequence，`✗` Lowercase/NFKC/`▁` 替换。
- C：C1 类切分、C2 数字整留、C3 空格贴后、C4 标点拖尾。
- D：BPE + byte fallback；`✓` 显式，`~` 词表隐含 256 bytes，`✗` WordPiece/Unigram/CTC。
- 排除：`B=✗且C3=✗`、`D=✗` 或非文本域；C1 无主流家族具备，不作排除项。

| 输入 | CLASS+mwn | Qwen2 | o200k/Llama3/Tekken/GLM/Phi | GPT-2/BART/OPT |
|---|---|---|---|---|
| `Hello你好` | `[Hello,你好]` | 整片 | 整片 | 整片 |
| `12345678` | 整片 | 逐字 | 3 位块 | 整片 |
| `U.S.` / `Hello,` / `No.` | 整片 | 碎 | 碎 | 碎 |
| `a  b` | `[a,␣␣b]` | `[a,␣,␣b]` | 同左 | 同左 |
| `Hello 你好` | `[Hello,␣你好]` | 相同 | 相同 | 相同 |
| `1️⃣` | 整片（gap 合并） | 碎 | 碎 | 碎 |

结论：C1/C4 为本方案独有；C2 仅 GPT-2 系对齐；C3 方向一致，但双空格以上差一段。

## 4. 保留、排除与历史 pattern 边界

**保留桶**：qwen-classic、qwen3.5+、llama3、o200k、tekken、gpt2-r50k、gptneo-bpe、bart-bpe、deepseek、whisper-bpe、bloom、modernbert、smollm、phi-4、glm、kimi、minimax、florence-2。共同点是骨架接近，仅 C1/C4 等行为不同。

**排除桶**：bert-wp 系、mpnet、contriever、nomic、distiluse 为 lowercase WordPiece；bge-m3/e5、t5-spm/marian-spm 为 Unigram/SPM；gemma、yi-1.5、llama2-spm 改写空格为 `▁`；wav2vec2-ctc 无 BPE；ESM/CLAP/Wan/Kronos/ACE 等非文本域。

实现口径须修正三处：

1. `hi␣`：`merged_with_next` 得 `['hi ']`；尾空格独立不可达。
2. `3.14`：本文 `CLASS_REGEX` 得 `['3.14']`，无拖尾变体得 `['3.','.14']`；与 `[3,.,14]` 要求互斥。字母拖尾与数字拆分不能在同一 pattern 下兼得。
3. 泰老/亚美尼亚/格鲁吉亚：三分支外脚本在 match 层是单 gap，可与相邻 match 合并。新增脚本分支只会把整段变成 match，不保证逐字符切分。

## 5. 最终下载集

下载集保留 **23 个家族**；这是审计口径，部分同架构的不同版本并列保留，不代表 23 个词表彼此独立。快照时条目均 `gated=False`。

| 家族 | 当前 repo | vocab |
|---|---|---:|
| qwen3.5+ | `Qwen/Qwen3.8-27B` | 248,077 |
| llama3 | `unsloth/Llama-3.3-70B-Instruct` | 128,256 |
| tekken | `mistralai/Mistral-Small-4-119B-2603` | 131,072 |
| deepseek | `deepseek-ai/DeepSeek-V4.1-Flash` | 129,280 |
| smollm | `HuggingFaceTB/SmolLM3-3B` | 128,256 |
| glm | `zai-org/GLM-5.3` | 154,856 |
| kimi | `moonshotai/Kimi-K3` | tiktoken |
| minimax | `MiniMaxAI/MiniMax-M3` | 200,061 |
| llama4 | `unsloth/Llama-4-Scout-17B-16E-Instruct` | 201,135 |
| mimo | `XiaomiMiMo/MiMo-V2.6-Pro-RL` | 151,675 |
| k2-horizon | `IFM/K2-Horizon-375B-A23B` | 250,624 |
| inkling | `thinkingmachines/Inkling` | 200,058 |
| nemotron3 | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | 131,072 |
| muse-glimmer | `meta-models/Muse-Glimmer-30B` | 202,048 |
| deepseek-v4-pro | `deepseek-ai/DeepSeek-V4-Pro-0813` | 129,280 |
| qwen-classic | `Qwen/Qwen2.5-7B-Instruct` | 151,665 |
| o200k | `openai/gpt-oss-120b` | 200,019 |
| gpt2 / bart | 下载脚本内配置 | 50,257 / 50,265 |
| whisper | 下载脚本内配置 | 51,865 |
| bloom | 下载脚本内配置 | 250,680 |
| modernbert | 下载脚本内配置 | 50,368 |
| phi | `microsoft/phi-4` | 100,352 |

关键下载限制：Kimi 无 `tokenizer.json`，需 `tiktoken.model`；MiniMax 分词器在 `tokenizer/` 子目录；Qwen3.8 相对 Qwen2.5 已扩容；Llama-3.3 与 3.1 同词表；SmolLM3 改用 Llama-3 词表。

下载脚本名保留历史 `16`，实际 `FAMILIES` 为 23 项；`ALLOW` 包含 chat template 与 `added_tokens.json`。下列命令依次刷新当前 10 个覆盖集、完整 23 家族；`--only` 不删除未列出的软链。从其目录执行：

```bash
HF_ENDPOINT=https://hf-mirror.com python -P download_landscape_16.py --force --only \
  bloom k2-horizon qwen3.5+ muse-glimmer minimax inkling mimo nemotron3 deepseek whisper
HF_ENDPOINT=https://hf-mirror.com python -P download_landscape_16.py --force
```

必须加 `-P`，避免仓库根 `tokenizers/` crate 目录遮蔽 PyPI 包。脚本先尝试 `Tokenizer.from_pretrained`（Kimi 可失败），再用 `snapshot_download(allow_patterns=ALLOW)` 建软链；实体只进 HF 缓存。当前 `landsurvey/` 是景观覆盖集 10 个，加 3 个字符级 BPE 候选，共 13 个软链。

## 6. Chat template 结论

- ChatML：qwen 系、phi、mimo、smollm、tekken、nemotron3。
- Llama header：llama3、llama4、smollm；其余使用 deepseek `｜▁｜`、minimax `ai_setting`、inkling `<|message_*|>`、k2-horizon `<|ifm|>`、o200k `<|channel|>` 等自有协议。
- gpt2、bart、bloom、whisper、modernbert、o200k 无 chat template。
- 词表/协议复用：`tekken == nemotron3`；`qwen-classic == mimo`，后者另加多模态 token。

## 7. 词表去重与独立族

对 22 个可解析家族（Kimi/tiktoken 除外）逐对比较 `model.vocab ∪ added_tokens`；全局有 **609,055** 个不同 token，无完整词表集合完全相同者，但有 7 组 Jaccard ≥0.90：

| 组 | 共同 token | 词表大小 | 实质 |
|---|---:|---|---|
| mimo ≡ qwen-classic | 151,665 | 151,675 / 151,665 | MiMo 用 Qwen2.5 词表 |
| deepseek ≡ deepseek-v4-pro | 129,271 | 129,280 / 129,280 | 同基座，9 个 specials 不同 |
| llama3 ≡ smollm | 128,246 | 128,256 / 128,256 | SmolLM3 用 Llama-3 |
| bart ⊃ gpt2 | 50,257 | 50,265 / 50,257 | GPT-2 加 8 个 specials |
| nemotron3 ≡ tekken | 131,054 | 131,072 / 131,072 | 同 Mistral tekken |
| inkling ≡ o200k | 199,999 | 200,058 / 200,019 | Inkling 用 o200k |
| llama4 ≈ muse-glimmer | 200,009 | 201,135 / 202,048 | J=0.984 |

强包含：`phi` 含于 llama3/smollm 99.9%、glm 99.0%、qwen-classic/mimo 98.8%；`glm` 含 qwen3.5+ 92.4%、qwen-classic/mimo 82.5%；`qwen3.5+` 含 qwen-classic 86.8%、llama3 90.3%；`minimax` 含 gpt2/bart 89.9%、modernbert 90.5%；`o200k` 含于 llama4 70.2%；Bloom 与最近者仅 J=0.29。

因此 23 家族归并为 **16 个独立词表族**。覆盖/去重统计应按 7 组处理；行为对照不可去重，因为 specials、normalizer、pre-tokenizer 与 merge rank 仍有差异。

## 8. 最小覆盖集（仅词表条目）

- **家族支配 ≥90%：10 个（推荐）**：`bloom`、`k2-horizon`、`qwen3.5+`、`muse-glimmer`、`minimax`、`inkling`、`mimo`、`nemotron3`、`deepseek`、`whisper`。
- **≥85%：8 个**：再去掉 `mimo`、`whisper`。
- **≥99%：14 个**：`bart`、`bloom`、`deepseek`、`glm`、`inkling`、`k2-horizon`、`llama3`、`mimo`、`minimax`、`modernbert`、`muse-glimmer`、`nemotron3`、`qwen3.5+`、`whisper`。

10 个集的覆盖明细：o200k ← inkling 100%；tekken ← nemotron3 100%；deepseek-v4-pro ← deepseek 100%；qwen-classic ← mimo 100%；llama4 ← muse-glimmer 99.4%；phi ← mimo 98.8%；glm ← qwen3.5+ 92.4%；bart/gpt2 ← k2-horizon 91.0%；modernbert ← k2-horizon 90.7%；llama3/smollm ← qwen3.5+ 90.3%。

token 级并集覆盖 609,055 个不同 token：**≥99% 需 12 个，≥99.9% 需 15 个，100% 需 20 个**。压缩来自等价复用、超大词表作公共 piece 容器、同源超集。条目覆盖仍不等于行为等价：库存/去重用 10 个；行为对照仍用 16 个独立族。
