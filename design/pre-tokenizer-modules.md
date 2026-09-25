# Pre-Tokenizer Modules

## 1. Runtime contract

Current encode path is byte-span based, not the legacy `PreTokenizedString` path. `PipelinePreTokenizer` in `tokenizers/tk-encode/src/tokenizer/pipeline/pre_tokenizer.rs` runs:

```text
Bert | Delimiter | Digits | FixedLength | Punctuation | Sequence | Split
UnicodeScripts (feature-gated) | Whitespace | WhitespaceSplit | None
```

`PreTokenizer::pre_tokenize` appends `bitcannon::Span` values to a reusable buffer. Each span is a half-open byte range in the normalized chunk passed to the pre-tokenizer; it must be in bounds and start/end on UTF-8 character boundaries. `Model::tokenize_spans` then slices each span. `Sequence` applies children in declaration order, rebasing child spans after every stage. `None` emits one span covering the whole chunk.

`tk-serialize` reads canonical `version: "2.0"` JSON. `tk-convert` performs the JSON-to-JSON upgrade for legacy `1.0` shapes before the canonical reader sees them.

## 2. Twelve configuration names, two compatibility lowerings

The historical/configuration surface has twelve names. Ten map directly to runtime variants. `ByteLevel` and `Metaspace` do **not** exist as `PipelinePreTokenizer` variants: `tk-convert` lowers them into other components.

| Configuration name | Runtime form | Key behavior / parameters |
|---|---|---|
| `BertPreTokenizer` | `Bert` | Drops whitespace and isolates each ASCII or Unicode punctuation character. No fields. |
| `ByteLevel` | none | Legacy BPE byte-alphabet marker. Sets model `byte_level: true`; `use_regex: true` (default) becomes a GPT-2 `Split`, `false` leaves no split (standalone slot becomes `null`; a `Sequence` member is removed). `add_prefix_space: true` is refused; `trim_offsets` is not a current encode parameter. |
| `CharDelimiterSplit` | `Delimiter` | Removes one delimiter character and keeps the runs between delimiters. Empty runs are dropped. `delimiter` is required. |
| `Digits` | `Digits` | `individual_digits: false` keeps numeric runs; `true` isolates each numeric character. Default is `false`. |
| `FixedLength` | `FixedLength` | Chunks by Unicode scalar count, not bytes. Runtime default is `5`; the canonical reader requires an explicit `length`; `0` emits the whole chunk. |
| `Metaspace` | none | Legacy tag lowered to `MetaspaceNormalizer` plus a delimiter `Split`; it is not a runtime pre-tokenizer variant. |
| `Punctuation` | `Punctuation` | Splits ASCII/Unicode punctuation only. `behavior` defaults to `Isolated` and accepts all five delimiter behaviors. |
| `Sequence` | `Sequence` | Ordered composition over current spans. Nested `Sequence` is rejected by the canonical reader. |
| `Split` | `Split` | Literal or regex `pattern`, `behavior` (default `Isolated`), and `invert` (default `false`). |
| `UnicodeScripts` | `UnicodeScripts` when enabled | Splits at effective Unicode script changes. No fields; requires the `unicode-scripts` feature. |
| `Whitespace` | `Whitespace` | Unicode-aware `\w+` / `[^\w\s]+` classes: drops whitespace and separates word runs from symbol/punctuation runs. |
| `WhitespaceSplit` | `WhitespaceSplit` | Drops Unicode whitespace runs and keeps every other run, so punctuation remains attached. |

`ByteLevel` is still meaningful as a model and decoder concern. Its byte mapping is applied to the BPE vocabulary/model, and a separate `ByteLevel` decoder (or the BPE byte-level decode path) restores bytes. `tokenizers/tk-encode/src/pre_tokenizers/byte_level.rs` only retains the legacy offset helper; it is not a runtime splitter. `Metaspace` likewise remains a decoder tag; its legacy pre-tokenizer form is split into normalizer plus splitter. Canonical `pre_tokenizer` accepts the ten non-`none` rows above; `ByteLevel` and `Metaspace` must be converted first.

## 3. Delimiter behavior

`Punctuation` and `Split` use `SplitDelimiterBehavior`. For `the-final--countdown`:

| Behavior | Emitted spans |
|---|---|
| `Removed` | `the`, `final`, `countdown` |
| `Isolated` | `the`, `-`, `final`, `-`, `-`, `countdown` |
| `MergedWithPrevious` | `the-`, `final-`, `-`, `countdown` |
| `MergedWithNext` | `the`, `-final`, `-`, `-countdown` |
| `Contiguous` | `the`, `-`, `final`, `--`, `countdown` |

`CharDelimiterSplit` is fixed to `Removed`. `Digits` maps `false`/`true` to contiguous/isolated numeric behavior. The Metaspace lowering uses `MergedWithNext` on its replacement delimiter. Span offsets are byte ranges relative to the normalized chunk; removed delimiters can leave gaps. They are not final model tokens.

## 4. Behavior notes and examples

### Whitespace family

- `Whitespace`: `Hey man!` → `Hey`, `man`, `!`. Use it when punctuation should be separated from words.
- `WhitespaceSplit`: `Hey man!` → `Hey`, `man!`. Use it as a first stage when another splitter should operate on whitespace-delimited runs.
- `BertPreTokenizer`: one runtime pass implements “drop whitespace, then isolate each punctuation character.” With a normalizer that surrounds CJK characters with spaces, `Hey friend! How are you` becomes `Hey`, `friend`, `!`, `How`, `are`, `you`. It does not itself insert those spaces.

### Character, script, and fixed-size rules

- `Punctuation` does not split whitespace. `Sequence([WhitespaceSplit, Punctuation])` turns `Hi, there!` into `Hi`, `,`, `there`, `!`; order is significant.
- `Digits`: `Hey 123 friend!` becomes `Hey `, `123`, ` friend!`, or `Hey `, `1`, `2`, `3`, ` friend!` with `individual_digits: true`.
- `FixedLength(length=5)`: `Hello world` → `Hello`, ` worl`, `d`. Chunks count characters while spans still report byte ranges.
- `UnicodeScripts`: `どこで生れ。Yes` → `どこで生れ`, `。`, `Yes`; `Apples are りんご 林檎` → `Apples are `, `りんご 林檎`. U+30FC maps to Han, Hiragana/Katakana map to Han, and spaces are neutral (`Any`).

### `Split` and `Sequence`

`SplitPattern` is `String` or `Regex`. A `String` is a literal byte search; a `Regex` is compiled by the system backend when available. `invert: false` makes matches the delimiters; `invert: true` makes non-matching regions delimit output. Recognized GPT-2 patterns use the native FSM and do not require a regex backend. Other regexes need the `fancy-regex` feature. `Sequence` preserves declaration order; DeepSeek's recognized three-`Split` shape also has a native fused fast path.

### Compatibility lowerings

`ByteLevel` and `Metaspace` must be described by their lowered form, not by a nonexistent runtime variant:

- `ByteLevel`: converter requires a BPE model. `use_regex: true` (default) emits `Split(Regex(GPT2), Isolated, false)` with the shared `bitcannon::regexes::GPT2` pattern (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`); `use_regex: false` leaves no split (standalone slot becomes `null`; a `Sequence` member is removed). In a `Sequence`, `ByteLevel` must be the last member. `add_prefix_space: true` is unsupported; `false` is accepted but not carried, and `trim_offsets` belongs to the historical offset/decoder path, not current encode output.
- `Metaspace`: `replacement` is exactly one character. The converter appends `MetaspaceNormalizer` (`replacement`, `prepend`, `drop_whitespace`) to the normalizer chain, then emits `Split(String(replacement), MergedWithNext, false)` when `split: true`; `split: false` leaves no pre-tokenizer. `always` prepends the marker at each word/sequence start, `first` only at sequence offset zero, and `never` never prepends; `always` is the default. Legacy `add_prefix_space` is normalized (`true` is ignored; `false` is valid only with `never`), and `str_rep` is discarded. `Sequence[WhitespaceSplit, Metaspace]` is lowered with `drop_whitespace: true`, and that form supports only `always` plus `split: true`. A `Metaspace` decoder remains a separate decoder component.

## 5. Selection guide

| Use case | Choose |
|---|---|
| BERT-style word/punctuation boundaries | `BertPreTokenizer` |
| Byte-alphabet BPE such as GPT-2/RoBERTa | `ByteLevel` legacy config; canonical model flag plus optional GPT-2 `Split` |
| Simple words and punctuation | `Whitespace` |
| Whitespace first, custom punctuation second | `Sequence([WhitespaceSplit, ...])` |
| SentencePiece space markers | `Metaspace` legacy config; canonical normalizer plus `Split` |
| Mixed-language script boundaries | `UnicodeScripts` |
| Preserve or isolate digit structure | `Digits` |
| Fixed-size character experiments | `FixedLength` |
| One-character data delimiter | `CharDelimiterSplit` |
| Custom literal or regex rule | `Split` |
| No split before model | `None`, or legacy `ByteLevel` with `use_regex: false` |

## 6. Limits and verification

- Pre-tokenizer boundaries are model input chunks. BPE merges, WordPiece matching, Unigram scoring, and added-token extraction still happen later.
- Normalization runs before pre-tokenization; added tokens can split input into separate chunks. Verify full-pipeline IDs and decode, not only isolated split text.
- Test empty input, repeated delimiters, all five `Punctuation`/`Split` behaviors, multibyte UTF-8, script changes, and byte-span boundaries. Current `PipelineTokenizer` output carries IDs/type IDs, not legacy offset objects.
- Runtime enum and span contract: `tokenizers/tk-encode/src/tokenizer/pipeline/pre_tokenizer.rs`.
- Canonical reader: `tokenizers/tk-serialize/src/from_json/pre_tokenizers.rs`; compatibility lowerings: `tokenizers/tk-convert/src/convert.rs`.
- Metaspace normalizer and decoder: `tokenizers/tk-encode/src/normalizers/metaspace.rs` and `tokenizers/tk-encode/src/decoders/metaspace.rs`.

From `tokenizers/`:

```bash
cargo test -p tk-encode
cargo test -p tk-encode <filter>
cargo test -p tk-serialize
cargo test -p tk-convert
```

Enable `unicode-scripts` or `fancy-regex` when validating those optional paths. Use `cargo test -p tk-serialize --features serialize` for canonical writer round trips.
