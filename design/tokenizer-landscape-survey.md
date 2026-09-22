# 主流分词器景观调查：20k 元数据全扫 + 与本方案相似度比较

> 日期 2026-09-22。目标：确定 Hub 上绝大多数模型微调自哪几个根，只下载这几个根的分词器；
> 再按本方案四件套（`added_tokens` 预注册 / `normalizer=None` / `CLASS_REGEX+mwn` 预分词 /
> `BPE+byte_fallback`，见 `uax29-sentence-pretokenizer.md`）打分，排除差异过大者。
> 全程未下载权重/blob：20k 用列表 API 元数据；个例验证仅用 `tokenizer_config.json`
> （KB 级）与 `tokenizer.json` 头部 Range（≤1MB）。

## 1. 方法

1. `GET /api/models?sort=downloads&direction=-1&limit=100` + `expand[]=tags,cardData,…`，
   `Link rel=next` 游标翻页（`skip>3000` 被 API 限死返回 400，必须走游标）。
   取 Top-20k，29.2 亿下载。脚本 `/tmp/opencode/fetch_cursor.py`（支持 `--resume`）。
2. 谱系归根：`tags` 内 `base_model:*` + `cardData.base_model` 取首父，样本内追链（≤10 跳），
   多父（merge）记首父。`analyze_meta.py`。
3. 根→分词器家族：启发式归并（`family_map.py`），组织级兜底 + `other:<id>` 单列。
4. 相似度验证：`tokenizer_config.json`（判 `tokenizer_class`/`do_lower_case`/文件存在）+
   `tokenizer.json` 头部 Range（判 `normalizer`/`pre_tokenizer`/`model`/`byte_fallback`）。
   `verify_tok.py` / `headfetch.py` / `headfetch2.py`。
5. 行为对照：Python `regex` 模块 + 本树 fold 精确镜像
   （`tk-encode/src/tokenizer/pipeline/pre_tokenizer.rs:389-410`，已逐行核对源码），
   同输入跑 `CLASS_REGEX` vs Qwen2/o200k/GPT-2 三 pattern。`pretok_cmp.py`。

## 2. 覆盖结论（N=20k）

- 去重 **12,171 个根**；按根截断 95% 下载需 **1,867 个**——根粒度不收敛，必须再聚家族。
- 聚得 **6,207 个家族**；95% 需 **430 个文本家族**（尾部 ~390 个单例：TTS 音色/OCR/分类器，
  各几百万下载）。文本家族占全量下载 **90.5%**。
- 经济截断：**Top-15 = 71.1%，Top-35 = 82.1%**（文本下载量口径）。

Top-35（家族 :: 下载量M :: 仓库数 :: canonical 根）：minilm-bertwp 410.7/70；
qwen-classic 335.5/2044；bert-wp 235.3/1212；qwen3.5+ 213.0/1648；gemma 115.3/1027；
bge-bertwp 97.0/34；wav2vec2-ctc 96.5/187；bge-m3 63.1/36；electra-wp 57.2/88；
clip-bpe 52.7/148；whisper-bpe 51.7/198；llama3 38.6/944；t5-spm 37.8/139；
mpnet 36.4/22；e5 35.7/59；minimax 35.4/131；deepseek 30.2/283；gptneo-bpe 24.9/130；
nomic 22.4/47；gpt2-r50k 22.0/88；Florence-2 21.8/145；glm 18.9/262；o200k 14.5/99；
ESM 14.4/150；smollm 10.1/89；kimi 9.8/61；modernbert 9.6/68；contriever 8.1/4；
marian-spm 7.7/95；mistral-spm 7.4/156；tekken 7.0/141；bart-bpe 6.4/26；yi 6.3/152。

## 3. 评分框架与排除线

- A `added_tokens` 预注册：文本系全有；CTC/纯视觉/音频/蛋白/音乐无 → 出局。
- B `normalizer=None`：✓直通 / ~`NFC`/空 Sequence / ✗ Lowercase/NFKC/▁替换。
- C 预分词：C1 类切分、C2 数字整留、C3 空格贴后、C4 标点拖尾。
- D `BPE+byte_fallback`：✓显式 / ~词表隐含 256 bytes / ✗ WordPiece/Unigram/CTC。
- 排除线：`B=✗且C3=✗`，或 `D=✗`，或非文本域。C1 无一家主流具备，不作排除项。

## 4. 实测行为对照（关键行）

| 输入 | CLASS+mwn | Qwen2 | o200k/Llama3/Tekken/GLM/Phi | GPT-2/BART/OPT |
|---|---|---|---|---|
| `Hello你好` | `[Hello,你好]` | 整片 | 整片 | 整片 |
| `12345678` | 整片 | 逐字 | 3 位块 | 整片 |
| `U.S.`/`Hello,`/`No.` | 整片 | 碎 | 碎 | 碎 |
| `a  b` | `[a,␣␣b]` | `[a,␣,␣b]` | 同左 | 同左 |
| `Hello 你好` | `[Hello,␣你好]` | 相同 | 相同 | 相同 |
| `1️⃣` | 整片（gap 合并） | 碎 | 碎 | 碎 |

即：C1/C4 为本方案独有；C2 仅 GPT-2 系对齐；C3 方向一致、双空格以上差一段。

## 5. 保留桶（骨架合，只差 C1/C4）

qwen-classic、qwen3.5+（NFC 仅合成 Accent，不折全角/大小写）、llama3、o200k、
tekken（含 Small-3.1，实测 o200k-pattern、`normalizer=null`，已并入）、gpt2-r50k、
gptneo-bpe、bart-bpe、deepseek（normalizer 为空 Sequence≡None，GPT 式 Split）、
whisper-bpe（null）、bloom（null，自研标点 Split）、modernbert（NFC + GPT-2 式
ByteLevel）、smollm（null，Digits 逐字）、phi-4（null，o200k-pattern，实测）、
glm（null，o200k-pattern，实测）、kimi（tiktoken 系，无 tokenizer.json，下载
`tiktoken.model`+config）、minimax（根目录无分词器，用 `tokenizer/` 子目录的
vocab+merges+json 三件套）、florence-2（null，纯 ByteLevel，边缘保留）。

## 6. 排除桶（均有 KB 级实证）

- bert-wp / minilm-bertwp / bge-bertwp / mpnet / contriever / nomic / distiluse：
  WordPiece + `do_lower_case=true`（config 实测；nomic/mpnet 为本次验证）。
- bge-m3 / e5：`XLMRobertaTokenizer`，Unigram。
- t5-spm / marian-spm：Unigram/SPM。
- gemma：`Replace(空格→▁)` + Metaspace（前期已验源码）。
- yi-1.5 / llama2-spm：`Replace(空格→▁)` + `Prepend▁`（头部 Range 实测）。
- electra-wp：WordPiece。clip-bpe：`do_lower_case=true`（实测），大小写保留 violation。
- wav2vec2-ctc：无 BPE/分词器。ESM/clap/Wan/Kronos/ACE 等：非文本域。

## 7. 本方案文档与本树实现的三处矛盾（必须先修）

本树 `split.rs` 不存在 §7 的 `class_regex_*` 测试（仅 3 个旧测试），以下按 §2 pattern +
本树 fold 为准：

1. **`hi␣`**：文档称 `[hi,' ']`；实现得 `['hi ']`（有/无拖尾皆合并）。`merged_with_next`
   下尾空格独立不可达。
2. **`3.14`**：文档 §58/§7 要求 `[3,.,14]`；§2 pattern 得 `['3.14']`，无拖尾变体得
   `['3.','.14']`——三值互斥。字母拖尾（`U.S.` 整片）与数字拆分在单一 pattern 下不可兼得。
3. **泰老/亚美尼亚/格鲁吉亚“已验分开”**：三分支外脚本在 match 层无分支，纯串是单 gap，
   会与相邻 match 合并（如 `aกก→[aกก]`）。分开只可能发生在 encode 层（字节不同）。
   文档须澄清断言层级，否则加脚本分支。

## 8. 最终下载集

### 8.1 原定 16 份（去重后）

qwen-classic、qwen3.5+、llama3（用 unsloth 免申请版）、o200k（gpt-oss）、tekken、
gpt2（+opt 的 vocab/merges）、bart、deepseek（V3/V4 到手比 hash）、whisper、bloom、
modernbert、smollm、phi、glm、kimi（tiktoken.model）、minimax（`tokenizer/` 子目录）。
同族多版本（Qwen2.5/3、DeepSeek V3/V4、Gemma3/4）下载后比 `vocab+merges` hash 定并。
MiniLM（下载第 1）因 lowercase 照样出局——排除只看骨架，不看人气。

### 8.2 更新至 23 份（2026-09-22）

原则：**家族 = 分词器架构；每个家族换成该厂商最新模型**的词表。全部 `gated=False`，
经 `hf-mirror.com` 下载，**无需 `HF_TOKEN`**。落地方式：`Tokenizer.from_pretrained`
触发下载（落 HF 缓存）→ `snapshot_download(allow_patterns=ALLOW)` 取快照 → 仓库内软链
`dataset/pretok_cmp/landsurvey/<family>` 指向
`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>`。真实文件仅在 HF 缓存，
仓库内无实体文件。脚本 `dataset/pretok_cmp/download_landscape_16.py`（`FAMILIES` 现 23 项，
`ALLOW` 已含 `chat_template*.jinja` / `chat_template.json` / `added_tokens.json`）。
运行**必须加 `-P`**（否则仓库根 Rust crate 目录 `tokenizers/` 会遮蔽已安装的 `tokenizers` 包）。

**A. 更新为厂商最新（8）**

| 家族 | 原 repo | 现 repo | vocab | 备注 |
|---|---|---|---|---|
| qwen3.5+ | `Qwen/Qwen3-8B` | `Qwen/Qwen3.8-27B` | 151,665 → **248,077** | 词表大幅扩容，确有变化 |
| llama3 | `unsloth/Llama-3.1-8B-Instruct` | `unsloth/Llama-3.3-70B-Instruct` | 128,256 | 与 3.1 **同词表**，仅对齐版本号 |
| tekken | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `mistralai/Mistral-Small-4-119B-2603` | 131,072 | 仍 tekken；新增音频/多模态与 ChatML token |
| deepseek | `deepseek-ai/DeepSeek-V3` | `deepseek-ai/DeepSeek-V4.1-Flash` | 129,280 | 128k 基座，特殊 token 集变 |
| smollm | `HuggingFaceTB/SmolLM2-1.7B` | `HuggingFaceTB/SmolLM3-3B` | 128,256 | 换用 Llama-3 系词表 |
| glm | `THUDM/glm-4-9b` | `zai-org/GLM-5.3` | 151,552 → **154,856** | 官方已由 THUDM 迁至 zai-org |
| kimi | `moonshotai/Kimi-K2-Instruct` | `moonshotai/Kimi-K3` | tiktoken | 仍无 `tokenizer.json` |
| minimax | `MiniMaxAI/MiniMax-Text-01` | `MiniMaxAI/MiniMax-M3` | **200,061** | H3 是视频/音频生成模型（非文本），取 M3 |

**B. 新增家族（7）**

| 家族 | repo | vocab | 备注 |
|---|---|---|---|
| llama4 | `unsloth/Llama-4-Scout-17B-16E-Instruct` | 201,135 | 新 200k 架构，独立于 llama3 |
| mimo | `XiaomiMiMo/MiMo-V2.6-Pro-RL` | 151,675 | 基座 `MiMo-V2.6-Pro` 门控 401，开放权重在 `-RL` |
| k2-horizon | `IFM/K2-Horizon-375B-A23B` | 250,624 | |
| inkling | `thinkingmachines/Inkling` | 200,058 | 含 `tiktoken/` 目录，根 `tokenizer.json` 可用 |
| nemotron3 | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | 131,072 | |
| muse-glimmer | `meta-models/Muse-Glimmer-30B` | 202,048 | |
| deepseek-v4-pro | `deepseek-ai/DeepSeek-V4-Pro-0813` | 129,280 | 与 `deepseek` **并行保留**（词表条目不同）|

**C. 保持不变（8）**：`qwen-classic`(`Qwen/Qwen2.5-7B-Instruct`，classic 架构末代)、
`o200k`(`openai/gpt-oss-120b`，OpenAI 最新开源权重)、`gpt2`、`bart`、`whisper`、`bloom`、
`modernbert`、`phi`(`microsoft/phi-4`，**无 phi-5，即最新文本版**)。

> **2026-09-22 收敛为覆盖集**：`landsurvey/` 已由上述 23 个软链**收敛为「覆盖集 10 个」**
> （见 §13）：`bloom`、`k2-horizon`、`qwen3.5+`、`muse-glimmer`、`minimax`、`inkling`、
> `mimo`、`nemotron3`、`deepseek`、`whisper`；其余 13 个软链已删除。删除的只是**软链**，
> 真实文件仍在 HF 缓存；需要完整 23 份时用
> `HF_ENDPOINT=https://hf-mirror.com python -P download_landscape_16.py --force` 一键重建。

## 9. 复现

- 元数据：`/tmp/opencode/meta_*.jsonl`（20k 条；/tmp 易失，长期存档需另拷）。
- 脚本：`fetch_cursor.py`（游标拉取）、`analyze_meta.py`（归根排名）、
  `family_map.py` + `step1.py`（家族聚合）、`pretok_cmp.py`（行为对照）、
  `verify_tok.py` / `headfetch.py` / `headfetch2.py`（KB 级验证）。
- 注意：`skip>3000` 的列表 API 返回 400，必须游标；单模型接口 `quote` 须保留 `/`
 （`safe="/"`），且一次只带一个 `expand[]`。

## 10. Chat Template 与特殊 token（23 家族，实测）

`chat_template` 列：`config` = `tokenizer_config.json` 内联；`jinja` = 额外带 `chat_template.jinja`。
「特殊/控制 token」为节选（同族的 `reserved`/`dummy`/`place_holder` 槽位已折叠）。

| 家族 | tokenizer_class | vocab | chat_template | 关键特殊/控制 token（节选） |
|---|---|---|---|---|
| qwen-classic | Qwen2Tokenizer | 151,665 | config | `<\|im_start\|> <\|im_end\|>`；`<\|vision_start/end/pad\|> <\|image_pad\|> <\|video_pad\|>`；`<tool_call> </tool_call>`；`<\|fim_prefix/middle/suffix/pad\|>`；`<\|repo_name\|> <\|file_sep\|>` |
| qwen3.5+ | Qwen2Tokenizer | 248,077 | config+jinja | 同上 + `<think> </think>` + `<\|tool_response\|>` |
| llama3 | PreTrainedTokenizerFast | 128,256 | config+jinja | `<\|begin_of_text\|> <\|end_of_text\|> <\|start_header_id\|> <\|end_header_id\|> <\|eot_id\|> <\|eom_id\|> <\|python_tag\|>` + `<\|reserved_special_token_N\|>` |
| llama4 | PreTrainedTokenizer | 201,135 | config+jinja | `<\|begin_of_text\|> <\|end_of_text\|> <\|header_start\|> <\|header_end\|> <\|eot\|> <\|python_start\|> <\|python_end\|> <\|finetune_right_pad\|>` + 大量 reserved（1,135 个 add）|
| o200k (gpt-oss) | PreTrainedTokenizerFast | 200,019 | 无 | `<\|startoftext\|> <\|endoftext\|> <\|return\|> <\|constrain\|> <\|channel\|> <\|start\|> <\|end\|> <\|message\|> <\|call\|> <\|endofprompt\|>` |
| tekken | TokenizersBackend | 131,072 | jinja | `[AVAILABLE_TOOLS] [/AVAILABLE_TOOLS] [TOOL_RESULTS] [TOOL_CALLS] [SYSTEM_PROMPT] [IMG] [IMG_BREAK] [IMG_END] [AUDIO] [BEGIN_AUDIO]`；`<\|im_start\|> <\|im_end\|>`；`<think> </think> <tool_call> </tool_call> <tool_response> </tool_response>`；`<SPECIAL_N>` |
| gpt2 | (GPT2) | 50,257 | 无 | `<\|endoftext\|>` |
| bart | — | 50,265 | 无 | `<s> </s> <pad> <unk> <mask>` |
| deepseek | PreTrainedTokenizerFast | 129,280 | 无 | `<｜begin▁of▁sentence｜> <｜end▁of▁sentence｜> <｜▁pad▁｜> <｜User｜> <｜Assistant｜> <｜System｜> <\|EOT\|>`；tool/FIM/repo；**1,223 个** `｜place▁holder▁no▁N｜` 占位槽 |
| deepseek-v4-pro | PreTrainedTokenizerFast | 129,280 | 无 | 同 128k 基座；**独有** `<｜table｜> <｜td｜> <｜tr｜> <｜image｜> <｜image2｜>`；1,217 个占位槽 |
| whisper | WhisperTokenizer | 51,865 | 无 | `<\|startoftranscript\|> <\|transcribe\|> <\|translate\|> <\|startoflm\|> <\|startofprev\|>` + 99 语言 tag |
| bloom | — | 250,680 | 无 | 自研标点 BPE（仅 4 个 add）|
| modernbert | PreTrainedTokenizerFast | 50,368 | 无 | `[CLS] [SEP] [PAD] [MASK] [UNK]` + `<\|padding\|> <\|endoftext\|>` |
| smollm | PreTrainedTokenizerFast | 128,256 | jinja | `<\|begin_of_text\|> <\|start_header_id\|> <\|eot_id\|>`；`<\|im_start\|> <\|im_end\|>`；`<think> </think>`；`<tool_call> </tool_call> <tool_response> </tool_response>` |
| phi | GPT2Tokenizer | 100,352 | config | `<\|im_start\|> <\|im_end\|> <\|im_sep\|> <\|endofprompt\|>`；`<\|fim_prefix/middle/suffix\|>`；96 个 `<\|dummy_N\|>` |
| glm | TokenizersBackend | 154,856 | jinja | `[MASK] [gMASK] [sMASK]`；`<\|system\|> <\|user\|> <\|assistant\|>`；`<\|begin_of_image/video/audio\|>`；`<\|code_prefix/middle/suffix\|>`；`<think> </think> <tool_call> </tool_call>` |
| kimi | TikTokenTokenizer | tiktoken | 无 | 无 `added_tokens` 暴露（`tiktoken.model`），需 `transformers.AutoTokenizer` |
| minimax | PreTrainedTokenizerFast | 200,061 | jinja | `<fim_prefix/middle/suffix/pad>`；`<function_call> <code_interpreter>`；图像/语音/视频（`]<]image[>[` 式编码）；`<think> </think> <tool_call> </tool_call> </mm:think>` |
| mimo | Qwen2Tokenizer | 151,675 | config+jinja | Qwen 全套 + `<\|mimo_video_start/end\|> <\|mimo_audio_start/end\|> <\|audio_pad\|> <\|mimo_audio_eod\|>` |
| k2-horizon | PreTrainedTokenizerFast | 250,624 | jinja | `<\|ifm\|begin_of_text\|> <\|ifm\|endoftext\|> <\|ifm\|im_start\|> <\|ifm\|im_end\|>`；`<\|begin_of_thought\|> <\|end_of_thought\|>`；header/FIM（626 个 add）|
| inkling | PreTrainedTokenizerFast | 200,058 | jinja | `<\|message_user\|> <\|message_model\|> <\|message_system\|> <\|message_tool\|>`；`<\|content_thinking\|> <\|content_image\|> <\|content_invoke_tool_json\|> <\|audio\|> <\|end_message\|>` |
| nemotron3 | PreTrainedTokenizerFast | 131,072 | jinja | **与 tekken 同 token 集**（`[AVAILABLE_TOOLS]` 系 + `<\|im_start\|> <\|im_end\|>` + `<think>`）|
| muse-glimmer | TokenizersBackend | 202,048 | jinja | `<\|begin_of_text\|> <\|end_of_text\|> <\|start\|> <\|message\|> <\|eot\|> <\|finetune_right_pad\|>` + 2,048 个 add（多为 reserved）|

要点：
- **三种模板体系**：ChatML（`<|im_start|>`：qwen 系 / phi / mimo / smollm / tekken / nemotron3）·
  Llama-3 header（`<|start_header_id|>`：llama3 / llama4 / smollm）· 自有
  （deepseek `｜▁｜`、minimax `ai_setting`、inkling `<|message_*|>`、k2-horizon `<|ifm|>`、o200k `<|channel|>`）。
- 基础/非对话模型（gpt2 / bart / bloom / whisper / modernbert / o200k）无 `chat_template`。
- 跨厂商「同 token 集」复用两处：`tekken == nemotron3`、`qwen-classic == mimo`（后者另加多模态 token）。

## 11. 词表重复与「独立词表族」

对 22 个可解析家族（kimi 为 tiktoken 系除外）做逐对集合比对（`model.vocab` ∪ `added_tokens`；
Jaccard = 交集/并集，「含」= 交集/min）。全局不同 token = **609,055**。

**无逐字节完全重复**，但有 7 组「事实重复」（Jaccard ≥ 0.90）：

| 组 | 共同 token | 大小 | 实质 |
|---|---|---|---|
| `mimo` ≡ `qwen-classic` | 151,665 | 151,675 / 151,665 | MiMo 直接用 Qwen2.5 词表（仅多 10）|
| `deepseek` ≡ `deepseek-v4-pro` | 129,271 | 129,280 / 129,280 | 同 128k 基座，9 个特殊 token 不同 |
| `llama3` ≡ `smollm` | 128,246 | 128,256 / 128,256 | SmolLM3 用 Llama-3 词表 |
| `bart` ≡ `gpt2` | 50,257 | 50,265 / 50,257 | gpt2 ⊂ bart（+8 特殊符）|
| `nemotron3` ≡ `tekken` | 131,054 | 131,072 / 131,072 | Nemotron-3-Ultra 用 Mistral tekken 词表 |
| `inkling` ≡ `o200k` | 199,999 | 200,058 / 200,019 | Inkling 用 GPT-4o 的 o200k 词表 |
| `llama4` ≈ `muse-glimmer` | 200,009 | 201,135 / 202,048 | J=0.984，Muse-Glimmer ≈ Llama-4 |

**强包含（同源子集/超集，非重复）**：`phi` ⊂ `llama3`/`smollm`（含 99.9%）、⊂ `glm`（99.0%）、
⊂ `qwen-classic`/`mimo`（98.8%）；`glm` ⊃ `qwen3.5+`（92.4%）、⊃ `qwen-classic`/`mimo`（82.5%）；
`qwen3.5+` ⊃ `qwen-classic`（86.8%）、⊃ `llama3`（90.3%）；`minimax` ⊃ `gpt2`/`bart`（89.9%）、
⊃ `modernbert`（90.5%）；`o200k` ⇒ `llama4`（70.2%）；`bloom` 与最近者仅 J=0.29。

**结论：23 个家族 → 16 个独立词表族**（7 组近重复各并 1；余 9 个独立单例：
`bloom`、`glm`、`k2-horizon`、`minimax`、`modernbert`、`phi`、`qwen3.5+`、`whisper`、`kimi`）。

> 影响：做**多样性/覆盖度**统计时须按上述 7 组去重；做**行为对照**时**不要**去重
> （差异恰在特殊 token 与 normalizer / pre_tokenizer 策略上）。

## 12. 谱系图

产物（均在本目录）：
- `tokenizer-lineage.svg` — 矢量关系图（浏览器/IDE 直接打开）
- `tokenizer-lineage.dot` — Graphviz 源码（有 `dot` 可 `dot -Tpng` 出图）
- `tokenizer-lineage.mmd` — Mermaid 源码
- `tokenizer_lineage_graph.py` — 生成脚本（零依赖，`python -P` 运行）

编码：方框=家族（数字=词表大小）；`≡` 实线=近同一（J≥0.95）；紫实线箭头=`⊂` 强包含（含≥95%）；
虚线=共享核心（含 70–95%）；配色 = base 蓝 / final 红 / 派生绿 / 共享核心灰 / 子集紫 / 超集橙 / 独立青。

**核心基础词表（根）**

| 类型 | 基础词表 | 大小 | 被复用情况 |
|---|---|---|---|
| 通用根 | GPT-2 byte-BPE | 50,257 | Bart 直接派生；ModernBERT 含 73.5%、MiniMax 含 89.9% |
| 通用根 | o200k（GPT-4o） | 200,019 | Inkling 原样复用；Llama-4 / Muse-Glimmer 含 70.2% |
| 生态根 | Llama-3 BPE | 128,256 | SmolLM3 原样复用；phi-4 为 99.9% 子集 |
| 生态根 | Qwen2.5 BPE | 151,665 | MiMo 原样复用；Qwen3.8 含 86.8% |
| 生态根 | tekken（Mistral） | 131,072 | Nemotron-3-Ultra 原样复用 |
| 生态根 | DeepSeek V3 基座 | 128,000 | V4.1-Flash / V4-Pro 在其上扩展 |
| 独立根 | BLOOM / Whisper / K2-Horizon / Kimi | 250,680 / 51,865 / 250,624 / tiktoken | 无派生 |

**基础词表 → 最终演化词表**

| 基础词表（根） | 最终演化词表（叶） | 幅度 |
|---|---|---|
| GPT-2 (50,257) | **MiniMax-M3 (200,061)** | +149,804 |
| o200k (200,019) | **Muse-Glimmer (202,048)**（经 Llama-4）| +2,029 |
| Llama-3 (128,256) | **SmolLM3 (128,256)** | 0（等价）※ phi-4 是裁剪非演化 |
| Qwen2.5 (151,665) | **Qwen3.8 / qwen3.5+ (248,077)** | +96,412 |
| tekken (131,072) | **Nemotron-3-Ultra (131,072)** | 0（等价）|
| DeepSeek V3 (128,000) | **DeepSeek-V4-Pro-0813 (129,280)** | +1,280 |

值得一提：`GLM-5.3` 是少见「汇合节点」——同时是 Qwen3.8（92.4%）与 Qwen2.5（82.5%）的超集；
`phi-4` 是「裁剪子集」——词表 ≈ Llama-3 ∩ Qwen2.5 的公共 token（99.9% ⊂ Llama-3）。

## 13. 最小覆盖集（用几个词表覆盖全部）

两种口径（覆盖 = 词表**条目清单**层面，非分词行为等价）：

**A. 家族级支配覆盖**（每个词表都有某选中词表 ≥t 包含）

- **≥90%：10 个** ← 推荐：
  `bloom`(250,680)、`k2-horizon`(250,624)、`qwen3.5+`(248,077)、`muse-glimmer`(202,048)、
  `minimax`(200,061)、`inkling`(200,058)、`mimo`(151,675)、`nemotron3`(131,072)、
  `deepseek`(129,280)、`whisper`(51,865)
- **≥85%：8 个**：再去掉 `mimo`、`whisper`
- **≥99%：14 个**：`bart` `bloom` `deepseek` `glm` `inkling` `k2-horizon` `llama3` `mimo`
  `minimax` `modernbert` `muse-glimmer` `nemotron3` `qwen3.5+` `whisper`

覆盖明细（≥90% 集）：`o200k ← inkling 100%`、`tekken ← nemotron3 100%`、
`deepseek-v4-pro ← deepseek 100%`、`qwen-classic ← mimo 100%`、`llama4 ← muse-glimmer 99.4%`、
`phi ← mimo 98.8%`、`glm ← qwen3.5+ 92.4%`、`bart`/`gpt2` ← `k2-horizon 91.0%`、
`modernbert ← k2-horizon 90.7%`、`llama3`/`smollm` ← `qwen3.5+ 90.3%`。

**B. token 级并集覆盖**（全局 609,055 个不同 token）

- **≥99%：12 个**；**≥99.9%：15 个**；**100%：20 个**。

**为什么能压到 10**：等价复用（1 顶 2）+ 超大词表充当「条目容器」（bloom / k2-horizon /
qwen3.5+ 体量最大，天然含大量跨语言公共 piece）+ 同源超集（qwen3.5+ ⊃ qwen-classic/glm/phi）。

**限定**：条目覆盖 ≠ 行为等价。即使 A 含 B 的 90% 条目，同句切分仍可能不同
（merge rank / normalizer / pre_tokenizer 各异）。→ 库存/去重统计用这 10 个；行为对照仍需 16 个独立族。

## 14. 本轮新增产物

- `dataset/pretok_cmp/download_landscape_16.py` — 下载脚本，`FAMILIES` 已更新至 23 项
- `dataset/pretok_cmp/landsurvey/*` — **覆盖集 10 个**软链（指向 HF 缓存快照；原 23 个已收敛，见 §8.2）
- 本目录：`tokenizer-lineage.svg` / `.dot` / `.mmd`、`tokenizer_lineage_graph.py`
- 顺带修复：`bloom` 的 `tokenizer.json` 原为**截断文件**，已重下补全（vocab 250,680）
