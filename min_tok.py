import argparse
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Sequence
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

CLASS_REGEX = (
    r"\p{White_Space}*"
    r"(?:"
    # CJK：只取连续汉字段，硬断点即断句，唯中点续接。
    # 硬断点（零他义）：。、…——（成双）、全角！？；，及换行。原拖尾写法
    # (?:[^\p{White_Space}\p{L}\p{N}]*[\p{Han}...]+)* 会跨标点把整句并成一个
    # pre-token，使 trainer 词表条目近乎全唯一、哈希表无限膨胀，中文侧吞吐崩到个位数
    # MiB/s 且内存持续爬升，故去拖尾。拉丁与数字分支不受影响。
    # 例外：中点 ·（U+00B7）・（U+30FB）･（U+FF65 半角）是译名间隔
    # （奥巴马·马 / オバマ･ケア），属词内不断，允许续接同类；其余标点一律断开。
    # · 罕见且只连短名，不会堆出长 run。
    # 谚文 \p{Hangul} 并入本类：韩文此前无分支、整行成 gap 独片；
    # 并入后韩文按空格散开（한국어 텍스트입니다 → 两片），与英文同机制；
    # 汉谚无空格相接时粘连（如 中文한국어 一片），韩文名中的 · 同样续接。
    r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}\u30FC]+"
    r"(?:[·・･][\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}\u30FC]+)*"
    r"|[\p{Latin}]+"
    r"(?:[^\p{White_Space}\p{L}\p{N}]*[\p{Latin}]+)*"
    r"|\p{N}+"
    r"(?:[^\p{White_Space}\p{L}]*\p{N}+)*"
    r")"
)

TOKEN_PROFILE = "both"

GENERIC_SPECIALS = [
    "<unk>", "<pad>", "<s>", "</s>", "<create>", "</create>", 
    "<|extra_0|>", "<|extra_1|>", "<|extra_2|>", "<|extra_3|>", "<|extra_4|>", "<|extra_5|>", "<|extra_6|>", "<|extra_7|>", "<|extra_8|>", "<|extra_9|>", "<|extra_10|>", "<|extra_11|>", "<|extra_12|>", "<|extra_13|>", "<|extra_14|>", "<|extra_15|>", "<|extra_16|>", "<|extra_17|>", "<|extra_18|>", "<|extra_19|>", "<|extra_20|>", "<|extra_21|>", "<|extra_22|>", "<|extra_23|>", "<|extra_24|>", "<|extra_25|>", "<|extra_26|>", "<|extra_27|>", "<|extra_28|>", "<|extra_29|>", "<|extra_30|>", "<|extra_31|>", "<|extra_32|>", "<|extra_33|>", "<|extra_34|>", "<|extra_35|>", "<|extra_36|>", "<|extra_37|>", "<|extra_38|>", "<|extra_39|>", "<|extra_40|>", "<|extra_41|>", "<|extra_42|>", "<|extra_43|>", "<|extra_44|>", "<|extra_45|>", "<|extra_46|>", "<|extra_47|>", "<|extra_48|>", "<|extra_49|>"
]

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

GENERIC_PROTOCOL_TOKENS = [
    "<think>", "</think>", "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<|extra_50|>", "<|extra_51|>", "<|extra_52|>", "<|extra_53|>", "<|extra_54|>", "<|extra_55|>", "<|extra_56|>", "<|extra_57|>", "<|extra_58|>", "<|extra_59|>", "<|extra_60|>", "<|extra_61|>", "<|extra_62|>", "<|extra_63|>", "<|extra_64|>", "<|extra_65|>", "<|extra_66|>", "<|extra_67|>", "<|extra_68|>", "<|extra_69|>", "<|extra_70|>", "<|extra_71|>", "<|extra_72|>", "<|extra_73|>", "<|extra_74|>", "<|extra_75|>", "<|extra_76|>", "<|extra_77|>", "<|extra_78|>", "<|extra_79|>", "<|extra_80|>", "<|extra_81|>", "<|extra_82|>", "<|extra_83|>", "<|extra_84|>", "<|extra_85|>", "<|extra_86|>", "<|extra_87|>", "<|extra_88|>", "<|extra_89|>", "<|extra_90|>", "<|extra_91|>", "<|extra_92|>", "<|extra_93|>", "<|extra_94|>", "<|extra_95|>", "<|extra_96|>", "<|extra_97|>", "<|extra_98|>", "<|extra_99|>"
]

# Qwen 工具/协议 token：Qwen3.8 tokenizer.json 中 special=false。
QWEN_PROTOCOL_TOKENS = [
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", 
]

# DeepSeek-V4.1 chat encoder 活跃 token：均为 tokenizer.json special=false。
DEEPSEEK_V4_1_CHAT_TOKENS = [
    "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<｜latest_reminder｜>",
    "｜DSML｜", "<｜deepseek_image｜>",
    "<｜action｜>", "<｜query｜>", "<｜authority｜>", "<｜domain｜>",
    "<｜title｜>", "<｜read_url｜>", 
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
    PROTOCOL_TOKENS = [*GENERIC_PROTOCOL_TOKENS, *QWEN_PROTOCOL_TOKENS]
elif TOKEN_PROFILE == "deepseek":  # 严格 DeepSeek-V4.1
    SPECIALS = [*GENERIC_SPECIALS, *DEEPSEEK_SPECIALS]
    PROTOCOL_TOKENS = [*GENERIC_PROTOCOL_TOKENS, *DEEPSEEK_PROTOCOL_TOKENS]
elif TOKEN_PROFILE == "both":  # 仅词表并集；chat renderer 仍须二选一
    SPECIALS = [*GENERIC_SPECIALS, *QWEN_SPECIALS, *DEEPSEEK_SPECIALS]
    PROTOCOL_TOKENS = [*GENERIC_PROTOCOL_TOKENS, *QWEN_PROTOCOL_TOKENS, *DEEPSEEK_PROTOCOL_TOKENS]
else:
    raise ValueError(f"unknown TOKEN_PROFILE: {TOKEN_PROFILE}")

SPECIALS = list(dict.fromkeys(SPECIALS))
PROTOCOL_TOKENS = list(dict.fromkeys(PROTOCOL_TOKENS))
BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))

CORPUS_PATH = Path("/path/to/your/corpus.txt")
ALPHABET_PATH = Path("/path/to/your/alphabet.json")  # 可选；仓库样本见 dataset/pretok_cmp/alphabet.json

# 131072 目标：--vocab-size 指最终 get_vocab() 总量（含 specials 与协议 token）。
# 实测 BpeTrainer.vocab_size 是总量上限（含 specials），故内部反推：
# trainer 预算 = target - len(PROTOCOL_TOKENS) - 256(byte)，训练后再追加两者，
# 无碰撞时最终总量精确等于 target，有碰撞则门禁 loud-fail。
TARGET_VOCAB_SIZE = 131072
DEFAULT_MIN_FREQUENCY = 5  # GB 级采样不用 2：过滤拼写噪声，省 trainer 内存
DEFAULT_TRAIN_GB = 32.0  # 64G 机默认采样量；小内存机降此值或升 min_frequency
DEFAULT_ALPHABET_GB = 2.0  # 字母表单遍扫描上限，与训练流同顺序前缀
TEXT_FIELD = "text"  # .jsonl/.json 行的文本字段
DEFAULT_GLOB = "*.txt"
LOG_EVERY_BYTES = 512 << 20  # U 盘 stall 可见：每 512MiB 打一次进度


def resolve_files(corpus, glob):
    """--corpus 可指单文件或目录（U 盘即插路径）；目录按 glob 递归收集并排序。"""
    path = Path(corpus)
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob(glob) if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files match {glob!r} under {corpus}")
    return files


def maybe_shuffle(files, shuffle, seed):
    if not shuffle:
        return files
    rng = random.Random(seed)
    return sorted(files, key=lambda _: rng.random())


def open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace", buffering=1 << 20)


def iter_lines(path, text_field, score_filter=None):
    """纯文本逐行；.jsonl/.json 抽 text_field；
    .parquet 用 pyarrow 按 batch 流读 text_field 列，可选 (score_field, threshold) 过滤。
    坏行/空行跳过并计数。"""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        field, threshold = score_filter or (None, None)
        cols = [text_field] if field in (None, text_field) else [text_field, field]
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        missing = [c for c in cols if c not in available]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}, available: {sorted(available)}")
        bad = 0
        for batch in pf.iter_batches(batch_size=8192, columns=cols):
            texts = batch.column(text_field).to_pylist()
            scores = batch.column(field).to_pylist() if field and field != text_field else None
            for i, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    bad += 1
                    continue
                if scores is not None and not (scores[i] is not None and scores[i] >= threshold):
                    bad += 1
                    continue
                yield text if text.endswith("\n") else text + "\n"
        if bad:
            print(f"[skip] {path}: {bad} empty/filtered rows", file=sys.stderr, flush=True)
        return
    is_jsonl = path.suffix in (".jsonl", ".json") or path.name.endswith(".jsonl.gz")
    bad = 0
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            if is_jsonl:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                text = obj.get(text_field) if isinstance(obj, dict) else None
                if not isinstance(text, str) or not text:
                    bad += 1
                    continue
                yield text if text.endswith("\n") else text + "\n"
            else:
                yield line
    if bad:
        print(f"[skip] {path}: {bad} bad jsonl lines", file=sys.stderr, flush=True)


def stream_corpus(files, budget_bytes, text_field, tag, stats, score_filter=None):
    """确定性顺序单遍流，字节预算封顶；顺序读对 U 盘友好，进度打到 stderr。"""
    consumed = 0
    yielded = 0
    opened = []
    start = time.time()
    next_log = LOG_EVERY_BYTES
    stop = False
    for path in files:
        if stop:
            break
        opened.append(str(path))
        for line in iter_lines(path, text_field, score_filter):
            size = len(line.encode("utf-8", errors="replace"))
            if consumed + size > budget_bytes:
                stop = True
                break
            consumed += size
            yielded += 1
            yield line
            if consumed >= next_log:
                dt = max(time.time() - start, 1e-9)
                print(
                    f"[{tag}] {consumed / 2**30:.2f} GiB, {yielded} lines, "
                    f"{consumed / dt / 2**20:.1f} MiB/s",
                    file=sys.stderr, flush=True,
                )
                next_log += LOG_EVERY_BYTES
    stats.update(
        gib=round(consumed / 2**30, 3), lines=yielded, files=len(opened),
        seconds=round(time.time() - start, 1),
    )
    print(f"[{tag}] done: {stats['gib']} GiB, {yielded} lines, {len(opened)} files",
          file=sys.stderr, flush=True)


def split_budget(total_bytes, weights):
    """按权切分字节预算，余数归最后一个源；BPE 只计词频，源间顺序无关。"""
    total_w = sum(weights)
    shares = [int(total_bytes * w / total_w) for w in weights]
    shares[-1] += total_bytes - sum(shares)
    return shares


def stream_sources(sources_files, shares, text_field, tag, per_source_stats, score_filter=None):
    """多源按份额顺序流；每源独立确定性文件序。"""
    for files, share, stats in zip(sources_files, shares, per_source_stats):
        yield from stream_corpus(files, share, text_field, f"{tag}[{stats['path']}]", stats,
                                 score_filter)


def build_initial_alphabet(sources_files, alphabet_path, budget_bytes, text_field,
                           weights, score_filter=None):
    chars = {chr(cp) for cp in range(0x20, 0x7F)} | set("\n\r\t")
    for start, end in (
        (0x00A0, 0x0100), (0x0300, 0x0370), (0x2000, 0x2070),
        (0x3000, 0x3100), (0xFE00, 0xFE10), (0xFF01, 0xFF60),
    ):
        chars.update(chr(cp) for cp in range(start, end))
    if alphabet_path and Path(alphabet_path).exists():
        chars.update(json.loads(Path(alphabet_path).read_text(encoding="utf-8")))
    shares = split_budget(budget_bytes, weights)
    stats = [{"path": s["path"]} for s in sources_files]
    for line in stream_sources([s["files"] for s in sources_files], shares,
                               text_field, "alphabet", stats, score_filter):
        chars.update(line)
    return (
        sorted(
            ch for ch in chars
            if len(ch) == 1
            and not 0xD800 <= ord(ch) <= 0xDFFF
            and not 0xFDD0 <= ord(ch) <= 0xFDEF
            and ord(ch) not in (0xFFFE, 0xFFFF)
        ),
        stats,
    )


ALPHABET_RANGES = (
    (0x00A0, 0x0100), (0x0300, 0x0370), (0x2000, 0x2070),
    (0x3000, 0x3100), (0xFE00, 0xFE10), (0xFF01, 0xFF60),
)


def is_scalar(value):
    return (
        len(value) == 1
        and not 0xD800 <= ord(value) <= 0xDFFF
        and not 0xFDD0 <= ord(value) <= 0xFDEF
        and ord(value) not in (0xFFFE, 0xFFFF)
    )


def design_alphabet():
    """基础词表字母表：固定码点区间单字符，不读语料，可重建。"""
    chars = {chr(cp) for cp in range(0x20, 0x7F)} | set("\n\r\t")
    for start, end in ALPHABET_RANGES:
        chars.update(chr(cp) for cp in range(start, end))
    return sorted(value for value in chars if is_scalar(value))


def _new_base_tokenizer(class_regex):
    tokenizer = Tokenizer.from_str(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": class_regex},
                    "behavior": "MergedWithNext",
                    "invert": False,
                },
                "post_processor": None,
                "decoder": {
                    "type": "Sequence",
                    "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}],
                },
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": "<unk>",
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "byte_fallback": True,
                    "ignore_merges": True,
                    "vocab": {},
                    "merges": [],
                },
            },
            ensure_ascii=False,
        )
    )
    tokenizer.post_processor = None
    tokenizer.normalizer = None
    return tokenizer


def build_base_tokenizer():
    """§1 基础词表：SPECIALS + design_alphabet 进 model.vocab，
    PROTOCOL_TOKENS 经 add_tokens 追加，BYTE_TOKENS 补齐，merges 为空。"""
    assert SPECIALS[0] == "<unk>"
    assert len(SPECIALS) == len(set(SPECIALS))
    assert len(PROTOCOL_TOKENS) == len(set(PROTOCOL_TOKENS))
    assert not set(SPECIALS).intersection(PROTOCOL_TOKENS)
    alphabet = design_alphabet()
    assert len(alphabet) == len(set(alphabet))
    assert all(is_scalar(value) for value in alphabet)

    # special ID 0 起连续；其后是 alphabet 单字符，均不与字节 token 碰撞。
    vocab = {token: index for index, token in enumerate(SPECIALS)}
    next_id = len(vocab)
    for char in alphabet:
        if char not in vocab:
            vocab[char] = next_id
            next_id += 1
    assert len(vocab) == next_id
    assert not set(vocab).intersection(BYTE_TOKENS)

    tok = _new_base_tokenizer(CLASS_REGEX)
    data = json.loads(tok.to_str())
    data["model"]["vocab"] = vocab
    tok = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
    tok.post_processor = None
    tok.normalizer = None

    # special token 直接注册；协议 token 用 add_tokens 注册，保持 special=false。
    for token in SPECIALS:
        tok.add_special_tokens([AddedToken(
            token, special=True, normalized=False, single_word=False,
            lstrip=False, rstrip=False,
        )])
    for token in PROTOCOL_TOKENS:
        tok.add_tokens([AddedToken(
            token, special=False, normalized=False, single_word=False,
            lstrip=False, rstrip=False,
        )])

    tok = inject_byte_tokens(tok)
    tok.decoder = Sequence([ByteFallback(), Fuse()])
    return tok


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


def verify_base(tokenizer):
    data = json.loads(tokenizer.to_str())
    assert data["normalizer"] is None
    assert data["post_processor"] is None
    assert data["model"]["byte_fallback"] is True
    assert data["model"]["merges"] == []
    pretok = data["pre_tokenizer"]
    assert pretok["type"] == "Split"
    assert pretok["behavior"] == "MergedWithNext"
    assert pretok["invert"] is False
    assert "sentence_breaks" not in pretok
    assert tokenizer.token_to_id("<unk>") == 0
    special_ids = [tokenizer.token_to_id(token) for token in SPECIALS]
    assert special_ids == list(range(len(SPECIALS)))
    for token in SPECIALS:
        added = tokenizer.get_added_tokens_decoder()[special_ids[SPECIALS.index(token)]]
        assert added.special is True and added.normalized is False
    for token in PROTOCOL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        assert token_id is not None
        added = tokenizer.get_added_tokens_decoder()[token_id]
        assert added.special is False and added.normalized is False
        assert not added.lstrip and not added.rstrip

    byte_vocab = {
        token for token in tokenizer.get_vocab()
        if re.fullmatch(r"<0x[0-9A-F]{2}>", token)
    }
    assert byte_vocab == set(BYTE_TOKENS)

    # 基础词表 merges 为空：CJK 汉字走 byte_fallback，decode 无损即可，
    # 不要求 offset 按字符连续（训练后 merges 覆盖再收紧此断言）。
    alphabet = set(design_alphabet())
    sample = "Hello 你好，世界 123 abc"
    enc = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(enc.ids, skip_special_tokens=False) == sample
    assert all(
        sample[start:end] == token
        for token, (start, end) in zip(enc.tokens, enc.offsets)
        if len(token) == 1 and token in alphabet
    )

    split_cases = {
        "Hello你好": ["Hello", "你好"],
        "abc123你好456": ["abc", "123", "你好", "456"],
        "3.14": ["3.14"],
        "U.S.": ["U.S."],
        "a  b": ["a", "  b"],
    }
    for text, expected in split_cases.items():
        got = [part for part, _ in tokenizer.pre_tokenizer.pre_tokenize_str(text)]
        assert got == expected, (text, got, expected)

    # Qwen/DeepSeek 协议 token 必须原子匹配，且 skip_special_tokens=True 仍保留。
    if TOKEN_PROFILE in ("qwen", "both"):
        qwen_tool_sample = (
            f"{QWEN_PROTOCOL_TOKENS[0]}get_weather{QWEN_PROTOCOL_TOKENS[1]}"
            f"{QWEN_PROTOCOL_TOKENS[2]}ok{QWEN_PROTOCOL_TOKENS[3]}"
        )
        qwen_enc = tokenizer.encode(qwen_tool_sample, add_special_tokens=False)
        assert all(
            tokenizer.token_to_id(token) in qwen_enc.ids
            for token in QWEN_PROTOCOL_TOKENS[:4]
        )
        assert tokenizer.decode(qwen_enc.ids, skip_special_tokens=True) == qwen_tool_sample

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
        deepseek_enc = tokenizer.encode(deepseek_tool_sample, add_special_tokens=False)
        assert all(
            tokenizer.token_to_id(token) in deepseek_enc.ids
            for token in ("<｜System｜>", "<｜Assistant｜>", "｜DSML｜")
        )
        assert tokenizer.decode(
            deepseek_enc.ids, skip_special_tokens=True
        ) == deepseek_tool_sample

    return {
        "trained": False,
        "contains_training_corpus": False,
        "token_profile": TOKEN_PROFILE,
        "counts": {
            "specials": len(SPECIALS),
            "protocol_tokens": len(PROTOCOL_TOKENS),
            "initial_alphabet": len(design_alphabet()),
            "byte_tokens": len(BYTE_TOKENS),
            "model_vocab": len(data["model"]["vocab"]),
            "merges": 0,
        },
    }


def trainer_vocab_size(target_vocab):
    """最终总量反推 trainer 预算；specials 含在预算内，protocol/byte 训后追加。"""
    return target_vocab - len(PROTOCOL_TOKENS) - len(BYTE_TOKENS)


def train_and_save(corpus, alphabet_path, output, target_vocab=TARGET_VOCAB_SIZE,
                   min_frequency=DEFAULT_MIN_FREQUENCY, train_bytes=int(DEFAULT_TRAIN_GB * 2**30),
                   alphabet_bytes=int(DEFAULT_ALPHABET_GB * 2**30), glob=DEFAULT_GLOB,
                   text_field=TEXT_FIELD, shuffle_files=False, seed=1337,
                   weights=None, score_filter=None):
    """§3 自训路径：采样流训练 + 字节补齐 + 严格门禁（需语料）。

    corpus 可多源（中英分目录），按 weights 切分字节预算后顺序流；
    字母表扫描同样按权覆盖各源，避免单语字母表漏字符。
    64G 机默认训 32GiB 采样；小内存机降 --train-gb 或升 --min-frequency。
    """
    if isinstance(corpus, (str, Path)):
        corpus = [corpus]
    weights = list(weights) if weights else [1.0] * len(corpus)
    assert len(weights) == len(corpus), "--weights 个数须与 --corpus 一致"
    sources = [
        {"path": str(c), "files": maybe_shuffle(resolve_files(c, glob), shuffle_files, seed + i)}
        for i, c in enumerate(corpus)
    ]
    initial_alphabet, alpha_stats = build_initial_alphabet(
        sources, alphabet_path, alphabet_bytes, text_field, weights, score_filter)
    assert all(len(ch) == 1 for ch in initial_alphabet)
    assert len(SPECIALS) == len(set(SPECIALS))
    assert len(PROTOCOL_TOKENS) == len(set(PROTOCOL_TOKENS))
    assert not set(SPECIALS).intersection(PROTOCOL_TOKENS)

    budget = trainer_vocab_size(target_vocab)
    assert budget > len(SPECIALS) + len(initial_alphabet), (
        f"trainer budget {budget} too small for {len(SPECIALS)} specials + "
        f"{len(initial_alphabet)} alphabet chars; lower --vocab-size is infeasible"
    )

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
        vocab_size=budget,
        min_frequency=min_frequency,
        special_tokens=SPECIALS,
        initial_alphabet=initial_alphabet,
        show_progress=True,
    )
    train_stats = [{"path": s["path"]} for s in sources]
    tok.train_from_iterator(
        stream_sources([s["files"] for s in sources], split_budget(train_bytes, weights),
                       text_field, "train", train_stats, score_filter), trainer)
    trained_size = len(tok.get_vocab())
    trained_pieces = set(tok.get_vocab())

    # Qwen/DeepSeek 的工具 token 实际是 special=false；不能用 add_special_tokens()。
    for token in PROTOCOL_TOKENS:
        tok.add_tokens([AddedToken(
            token, special=False, normalized=False, single_word=False,
            lstrip=False, rstrip=False,
        )])
    protocol_new = len(tok.get_vocab()) - trained_size

    tok = inject_byte_tokens(tok)
    tok.decoder = Sequence([ByteFallback(), Fuse()])
    tok.save(str(output))

    reloaded = Tokenizer.from_file(str(output))
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

    final_size = len(data["model"]["vocab"])
    total_size = len(reloaded.get_vocab())
    byte_new = sum(1 for b in BYTE_TOKENS if b not in trained_pieces)
    assert total_size == target_vocab, (
        f"total vocab {total_size} != target {target_vocab}: "
        f"trained={trained_size} protocol_new={protocol_new} byte_new={byte_new}; "
        "语料含字面 <0xHH>/协议串发生碰撞，或语料太小致 trainer 未触顶——"
        "检查 manifest 后调 --vocab-size 或换语料"
    )

    manifest = {
        "trained": True,
        "token_profile": TOKEN_PROFILE,
        "target_vocab": target_vocab,
        "trainer_budget": budget,
        "min_frequency": min_frequency,
        "train_stream": train_stats,
        "alphabet_stream": alpha_stats,
        "sources": [
            {"path": s["path"], "weight": w, "n_files": len(s["files"])}
            for s, w in zip(sources, weights)
        ],
        "glob": glob,
        "score_filter": list(score_filter) if score_filter else None,
        "shuffle_files": shuffle_files,
        "seed": seed,
        "counts": {
            "specials": len(SPECIALS),
            "protocol_tokens": len(PROTOCOL_TOKENS),
            "protocol_new": protocol_new,
            "initial_alphabet": len(initial_alphabet),
            "trained_vocab": trained_size,
            "byte_new": byte_new,
            "model_vocab": final_size,
            "total_vocab": total_size,
            "merges": len(data["model"]["merges"]),
        },
    }
    manifest_path = output.with_name(output.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="§1 基础词表生成 / §3 自训")
    parser.add_argument("--mode", choices=("base", "train"), default="base")
    parser.add_argument("--output", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--corpus", type=Path, action="append", default=None,
                        help="可多次指定多源（如中英分目录）；默认 CORPUS_PATH")
    parser.add_argument("--weights", default=None,
                        help="与 --corpus 一一对应的逗号权重（如 96,81），默认均分")
    parser.add_argument("--alphabet", type=Path, default=ALPHABET_PATH)
    parser.add_argument("--vocab-size", type=int, default=TARGET_VOCAB_SIZE,
                        help="最终 get_vocab() 总量（默认 131072）")
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument("--train-gb", type=float, default=DEFAULT_TRAIN_GB,
                        help="训练流字节预算 GiB（默认 32；小内存机调小）")
    parser.add_argument("--alphabet-gb", type=float, default=DEFAULT_ALPHABET_GB,
                        help="字母表扫描字节预算 GiB（默认 2）")
    parser.add_argument("--glob", default=DEFAULT_GLOB,
                        help="--corpus 为目录时的递归匹配（默认 *.txt）")
    parser.add_argument("--text-field", default=TEXT_FIELD,
                        help=".jsonl/.json 行的文本字段（默认 text）")
    parser.add_argument("--shuffle-files", action="store_true",
                        help="按 seed 打乱文件顺序（默认排序顺序流式）")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-score", type=float, default=None,
                        help="parquet 行过滤下限（默认不过滤）")
    parser.add_argument("--score-field", default="score",
                        help="parquet 分数字段（默认 score）")
    args = parser.parse_args()
    if args.mode == "base":
        tok = build_base_tokenizer()
        tok.save(str(args.output))
        print(json.dumps(verify_base(tok), ensure_ascii=False, indent=2))
    else:
        corpuses = args.corpus or [CORPUS_PATH]
        weights = [float(w) for w in args.weights.split(",")] if args.weights else None
        score_filter = (args.score_field, args.min_score) if args.min_score is not None else None
        train_and_save(
            corpuses, args.alphabet, args.output,
            target_vocab=args.vocab_size, min_frequency=args.min_frequency,
            train_bytes=int(args.train_gb * 2**30),
            alphabet_bytes=int(args.alphabet_gb * 2**30), glob=args.glob,
            text_field=args.text_field, shuffle_files=args.shuffle_files,
            seed=args.seed, weights=weights, score_filter=score_filter,
        )


if __name__ == "__main__":
    main()
