# Pre-Tokenizer Modules

## 1. Overview

The tokenizer library provides 12 pre-tokenizer modules, all implementing the `PreTokenizer` trait defined in `tokenizers/src/tokenizer/mod.rs`:

```rust
pub trait PreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()>;
}
```

Each module transforms a `PreTokenizedString` (which wraps raw text and tracks splits with offset mappings) by splitting it into smaller tokens according to its specific rules.

All 12 modules are re-exported via `PreTokenizerWrapper`, an enum that enables JSON deserialization from tokenizer config files.

---

## 2. Module Reference

### 2.1 Whitespace

- **File**: `tokenizers/src/pre_tokenizers/whitespace.rs:10-29`
- **Regex Pattern**: `\w+|[^\w\s]+`
- **Behavior**: Splits on word boundaries — matches contiguous alphanumeric sequences or isolated punctuation symbols. Whitespace between matches is discarded.

**Example**: `"Hey man!"` → `[("Hey",0,3), ("man",4,7), ("!",7,8)]`

**Implementation**: Uses `LazyLock<Regex>` for compiled regex caching. Applies regex via `Invert` pattern wrapper so matching regions become split points.

---

### 2.2 WhitespaceSplit

- **File**: `tokenizers/src/pre_tokenizers/whitespace.rs:31-41`
- **Predicate**: `char::is_whitespace`
- **Behavior**: Splits at every whitespace character, discarding the separator. Unlike `Whitespace`, punctuation stays attached to adjacent words.

**Example**: `"Hey man!"` → `[("Hey",0,3), ("man!",4,8)]`

**Difference from Whitespace**: Whitespace produces finer-grained splits by also isolating punctuation.

---

### 2.3 BertPreTokenizer

- **File**: `tokenizers/src/pre_tokenizers/bert.rs:9-18`
- **Algorithm**: Two-phase sequential splitting:
  1. Split on whitespace (`char::is_whitespace`) with `Removed` behavior
  2. Split on punctuation (`is_bert_punc`) with `Isolated` behavior

```rust
impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))?;
        pretokenized.split(|_, s| s.split(is_bert_punc, SplitDelimiterBehavior::Isolated))
    }
}
```

**Punctuation detection**: `char::is_ascii_punctuation(&x) || x.is_punctuation()`

**Chinese handling**: When combined with a normalizer that inserts spaces around CJK characters, this produces per-character tokenization for Chinese while keeping Latin words intact.

**Example**: `"Hey friend! How are you"` → `[("Hey"), ("friend"), ("!"), ("How"), ("are"), ("you")]`

---

### 2.4 ByteLevel

- **File**: `tokenizers/src/pre_tokenizers/byte_level.rs:51-148`
- **Lines of code**: 593 (includes Decoder implementation)
- **Also implements**: `Decoder` trait

**Configuration struct**:

```rust
pub struct ByteLevel {
    pub add_prefix_space: bool,   // Prepend space to first word
    pub trim_offsets: bool,       // Trim whitespace from encoding offsets
    pub use_regex: bool,          // Enable GPT-2 style regex splitting
}
```

**GPT-2 Regex** (line 44):

```regex
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

**Byte-level conversion** (`bytes_char()`, lines 15-39): Maps all 256 byte values to unique Unicode codepoints:
- Bytes 0-255 present in ASCII ranges map directly
- Remaining bytes map to `0x100 + n` where n counts missing bytes

This ensures every possible UTF-8 input can be represented as a sequence of byte-level tokens, preventing unknown token errors.

**Two-stage processing**:
1. `split()`: Apply regex to break text into initial tokens
2. `normalize()`: Convert each token's UTF-8 bytes to their byte-level Unicode equivalents

**Applicable models**: GPT-2, GPT-3, RoBERTa, DeBERTa, BPE-based models requiring byte-alphabet coverage.

---

### 2.5 Metaspace

- **File**: `tokenizers/src/pre_tokenizers/metaspace.rs:22-148`
- **Also implements**: `Decoder` trait
- **Default**: replacement char = `▁` (U+2581, low one-em space), prepend_scheme = `Always`, split = `true`

**Configuration enum**:

```rust
pub enum PrependScheme {
    First,   // Add prefix only to first split
    Never,   // No prefix prepending
    Always,  // Prefix to every split
}
```

**Algorithm**:
1. Replace all `' '` characters with `str_rep` (the replacement meta character)
2. Prepend `str_rep` based on `prepend_scheme`:
   - `Always`: If text doesn't start with replacement char, prepend it
   - `First`: Only if at original string position 0
   - `Never`: Skip
3. If `split = true`, split on the replacement character using `MergedWithNext` behavior

**Decoding logic** (`decode_chain`, lines 150-173): Converts meta characters back to spaces, preserving the first-space policy.

**JSON serialization**:

```json
{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}
```

**Multiple spaces handling**: Consecutive spaces produce empty meta-only tokens, which decode to multiple spaces.

**Use case**: SentencePiece-style tokenization (Unigram, BPE).

---

### 2.6 CharDelimiterSplit

- **File**: `tokenizers/src/pre_tokenizers/delimiter.rs:6-26`
- **Simplest module**: Single field `delimiter: char`

**Implementation**:

```rust
impl PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(self.delimiter, SplitDelimiterBehavior::Removed)
        })
    }
}
```

**Note**: Hardcoded to `Removed` behavior — no configurable merge/isolated modes.

**Use case**: CSV, TSV, or other simple delimiter-separated data.

---

### 2.7 Digits

- **File**: `tokenizers/src/pre_tokenizers/digits.rs:6-39`

**Configuration**:

```rust
pub struct Digits {
    pub individual_digits: bool,  // false by default
}
```

**Behavior toggle**:
- `individual_digits = false`: `SplitDelimiterBehavior::Contiguous` — consecutive digits merged into single token
- `individual_digits = true`: `SplitDelimiterBehavior::Isolated` — each digit becomes its own token

**Examples**:
- `false`: `"price 123 dollars"` → `[(" price "), ("123"), (" dollars")]`
- `true`: `"price 123 dollars"` → `[(" price "), ("1"), ("2"), ("3"), (" dollars")]`

**Use case**: Models sensitive to numeric content structure (e.g., time/date parsing, numerical reasoning).

---

### 2.8 FixedLength

- **File**: `tokenizers/src/pre_tokenizers/fixed_length.rs:7-50`

**Configuration**:

```rust
pub struct FixedLength {
    #[serde(default = "default_length")]  // default: 5
    pub length: usize,
}
```

**Algorithm**: Iterates over `char_indices()`, chunks by `length`, slices each chunk from the normalized string.

**UTF-8 safe**: Uses `char.len_utf8()` for correct byte-position calculation. Handles multi-byte characters correctly.

**Example** (length=5):
| Input | Output |
|-------|--------|
| `"Hello world"` | `[("Hello",0,5), (" worl",5,10), ("d",10,11)]` |
| `"Hello 👋 world"` | `[("Hel",0,3), ("lo ",3,6), ("👋 w",6,12), ("orl",12,15), ("d",15,16)]` |

**Empty input**: Returns empty splits.

**Use case**: Experiments requiring fixed-length segments, legacy model compatibility.

---

### 2.9 Punctuation

- **File**: `tokenizers/src/pre_tokenizers/punctuation.rs:11-38`

**Configuration**:

```rust
pub struct Punctuation {
    #[serde(default = "default_split")]  // Isolated by default
    pub behavior: SplitDelimiterBehavior,
}
```

**Behavior options**: `Removed`, `Isolated` (default), `MergedWithPrevious`, `MergedWithNext`, `Contiguous`

**Punctuation detection**: `char::is_ascii_punctuation(&x) || x.is_punctuation()`

**Example** (default `Isolated`):
`"Hey friend!     How are you?!?"` → `[("Hey friend"), ("!"), ("     How are you"), ("?"), ("!"), ("?")]`

**Comparison to BertPreTokenizer**: Single-pass punctuation isolation. Does not also split on whitespace. Often used in combination with another pre-tokenizer via `Sequence`.

---

### 2.10 Sequence

- **File**: `tokenizers/src/pre_tokenizers/sequence.rs:6-46`

**Purpose**: Compose multiple pre-tokenizers into a pipeline.

**Implementation**:

```rust
impl PreTokenizer for Sequence {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        for pretokenizer in &self.pretokenizers {
            pretokenizer.pre_tokenize(pretokenized)?;
        }
        Ok(())
    }
}
```

**Iterator support**: Implements `IntoIterator`, `AsRef<[PreTokenizerWrapper]>`, `AsMut<[PreTokenizerWrapper]>` for introspection.

**JSON format** (used by GPT-2/SentencePiece models):

```json
{
  "type": "Sequence",
  "pretokenizers": [
    {"type": "WhitespaceSplit"},
    {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}
  ]
}
```

**Typical usage**: Combine whitespace splitting with metaspace replacement for SentencePiece-compatible tokenization.

---

### 2.11 Split

- **File**: `tokenizers/src/pre_tokenizers/split.rs:8-104`

**Most flexible module**: Supports both literal strings and regex patterns with configurable behavior and inversion.

**Pattern types**:

```rust
pub enum SplitPattern {
    String(String),  // Literal match
    Regex(String),   // Compiled regex
}
```

**Configuration**:

```rust
pub struct Split {
    pub pattern: SplitPattern,
    #[serde(skip)]
    pub regex: SysRegex,         // Compiled regex instance
    pub behavior: SplitDelimiterBehavior,
    pub invert: bool,            // Invert matching (split on non-matching parts)
}
```

**Behavior table**:

| Behavior | Separator treatment | Example: `"how are you"` on space |
|----------|-------------------|----------------------------------|
| `Removed` | Discarded entirely | `[("how"), ("are"), ("you")]` |
| `Isolated` | Becomes own token | `[("how"), (" "), ("are"), (" "), ("you")]` |
| `MergedWithPrevious` | Appended to left | `[("how "), ("are "), ("you")]` |
| `MergedWithNext` | Prepended to right | `[("how"), (" are"), (" you")]` |
| `Contiguous` | Multiple separators merged | `[("how"), (" "), ("are"), (" "), ("you")]` |

**Invert mode**: When `invert = true`, splits on regions NOT matched by the pattern. Equivalent to `Invert(regex)` wrapping.

**Regex vs String**: Both compile to `SysRegex` internally. String literals are escaped via `regex::escape()`.

**JSON examples**:

```json
// Regex pattern
{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Removed","invert":false}
// String pattern with invert
{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}
```

**Use case**: Custom tokenization rules, language-specific splitting, testing/debugging.

---

### 2.12 UnicodeScripts

- **File**: `tokenizers/src/pre_tokenizers/unicode_scripts/pre_tokenizer.rs:1-77`
- **Supporting file**: `scripts.rs` (~2000 lines, covers 60+ Unicode scripts)

**Purpose**: Split text at script boundaries — whenever the Unicode property changes, create a new split.

**Script database** (`scripts.rs`): Auto-generated from SentencePiece's `gen_unicode_scripts_code.pl`. Maps each Unicode codepoint range to a `Script` variant. Covers: Latin, Greek, Cyrillic, Armenian, Hebrew, Arabic, Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Thai, Lao, Tibetan, Han, Hangul, Hiragana, Katakana, Ethiopic, Georgian, Cherokee, CanadianAboriginal, and many more.

**Fixed script mapping** (`fixed_script()`, lines 25-38):

```rust
fn fixed_script(c: char) -> Script {
    let raw_script = get_script(c);
    if c as u32 == 0x30FC { Script::Han }           // Long mark → Han
    else if c == ' ' { Script::Any }                 // Spaces → Any
    else {
        match raw_script {
            Script::Hiragana => Script::Han,         // Hiragana merged with Han
            Script::Katakana => Script::Han,         // Katakana merged with Han
            script => script,
        }
    }
}
```

**Algorithm**:
1. Iterate chars, determine script for each
2. Record split positions where script changes (excluding `Script::Any`)
3. Append end-of-string as final split point
4. Slice normalized string at each boundary pair

**Space handling**: Spaces (`Script::Any`) are always included within whichever script region they appear in — they never trigger splits.

**Example**:
| Input | Output |
|-------|--------|
| `"どこで生れ。Yes"` | `[("どこで生れ",0,15), ("。",15,18), ("Yes",18,21)]` |
| `"Apples are りんご 林檎"` | `[("Apples are ",0,11), ("りんご 林檎",11,27)]` |

**Use case**: Mixed-language text (CJK + Latin), Japanese preprocessing, multilingual NLP.

---

## 3. Common Patterns Across All Modules

### 3.1 Serialization Support

Every module uses the `impl_serde_type!` macro attribute, generating `Serialize` and `Deserialize` implementations. The macro adds:
- `#[derive(Serialize, Deserialize)]`
- `#[serde(tag = "type")]` for tag-based polymorphic deserialization

### 3.2 Splitting Interface

All modules call `pretokenized.split(fn)` which takes a closure accepting `(index: usize, NormalizedString)` and returning `Result<Vec<Split>>`. This closure receives ownership of each split's normalized string and returns new splits to replace it.

### 3.3 SplitDelimiterBehavior Enum

```rust
pub enum SplitDelimiterBehavior {
    Removed,           // Drop the delimiter entirely
    Isolated,          // Delimiter becomes its own token
    MergedWithPrevious, // Delimiter appended to preceding token
    MergedWithNext,     // Delimiter prepended to following token
    Contiguous,         // Consecutive delimiters form one token
}
```

### 3.4 Offset Tracking

All modules preserve offset mappings through the `NormalizedString` type. The `PreTokenizedString` supports two referentials:
- `OffsetReferential::Original` — positions in the raw input string
- `OffsetReferential::Normalized` — positions after normalization transformations

Three offset types: `Byte`, `Char`, `None`.

---

## 4. Design Decisions

### 4.1 Trait Simplicity

The `PreTokenizer` trait has a single method taking a mutable reference. No associated types, no generics. Concrete state is stored in the struct fields. This keeps the trait object-friendly and avoids complex type erasure.

### 4.2 Sequence Composition

Rather than providing builder-pattern APIs for complex pipelines, the library uses explicit composition via `Sequence`. Each pre-tokenizer transforms the `PreTokenizedString` in place, accumulating splits progressively. Later stages operate on earlier stages' output.

### 4.3 ByteLevel Dual Role

`ByteLevel` uniquely implements both `PreTokenizer` and `Decoder`. This reflects how byte-level tokenization requires knowledge of the decoding process — byte-to-char maps must be consistent in both directions. Other pre-tokenizers are unidirectional.

### 4.4 Metaspace Symmetry

Like `ByteLevel`, `Metaspace` implements `Decoder`. Its encode-decode round-trip handles:
- Space→meta conversion during encoding
- Meta→space restoration during decoding
- First-space policy across token boundaries

### 4.5 UnicodeScripts Performance

The `scripts.rs` file contains ~2000 lines of `match` arms covering the full Unicode BMP (0x0000–0xFFFF). While verbose, lookup is O(1) via direct codepoint matching. The generated file prioritizes correctness over size.

---

## 5. File Structure Summary

| Module | File | Line Count |
|--------|------|------------|
| Mod root | `pre_tokenizers/mod.rs` | 332 |
| Whitespace / WhitespaceSplit | `pre_tokenizers/whitespace.rs` | 105 |
| BertPreTokenizer | `pre_tokenizers/bert.rs` | 82 |
| ByteLevel | `pre_tokenizers/byte_level.rs` | 593 |
| CharDelimiterSplit | `pre_tokenizers/delimiter.rs` | 26 |
| Digits | `pre_tokenizers/digits.rs` | 102 |
| FixedLength | `pre_tokenizers/fixed_length.rs` | 122 |
| Metaspace | `pre_tokenizers/metaspace.rs` | 370 |
| Punctuation | `pre_tokenizers/punctuation.rs` | 83 |
| Sequence | `pre_tokenizers/sequence.rs` | 82 |
| Split | `pre_tokenizers/split.rs` | 253 |
| UnicodeScripts | `pre_tokenizers/unicode_scripts/` | ~2500 total |
| **Total source** | | **~4000** |

---

## 6. Module Selection Guide

| Use Case | Recommended PreTokenizer(s) |
|----------|---------------------------|
| BERT-family models | `BertPreTokenizer` |
| GPT/RoBERTa models | `ByteLevel` (or `Sequence([WhitespaceSplit, Metaspace])`) |
| Simple English text | `Whitespace` |
| CSV/data preprocessing | `CharDelimiterSplit` |
| Multilingual Japanese | `UnicodeScripts` |
| Number-sensitive tasks | `Digits(individual_digits=true)` |
| Custom regex splitting | `Split` |
| SentencePiece compatibility | `Sequence([WhitespaceSplit, Metaspace])` |
| Fixed-length experiments | `FixedLength(n)` |
| Combining strategies | `Sequence([...])` |
