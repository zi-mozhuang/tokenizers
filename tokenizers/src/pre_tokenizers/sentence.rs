use crate::tokenizer::{normalizer::Range, PreTokenizedString, PreTokenizer, Result};
use crate::utils::macro_rules_attribute;
use unicode_segmentation::UnicodeSegmentation;

/// Sentence-level pre-tokenizer implementing Unicode UAX #29 sentence boundaries.
///
/// Segments text with `unicode_segmentation`'s `split_sentence_bound_indices`,
/// a complete implementation of the UAX #29 default (root) sentence break rules
/// (SB1-SB16), and emits each sentence segment as one split.
///
/// Properties:
/// - **Lossless**: it never adds, removes or modifies any character. The
///   concatenation of all splits is always identical to the input.
/// - **Standard whitespace attribution**: following SB11, a boundary is placed
///   after `SATerm Close* Sp*`, so inter-sentence whitespace belongs to the
///   *end of the preceding sentence* (e.g. `"Hello.  World"` ->
///   `["Hello.  ", "World"]`).
/// - **No intra-sentence splits**: abbreviations (`Mr.`, `e.g.`), quotes,
///   parentheses, numbers and grapheme clusters (combining marks, ZWJ emoji)
///   are never split apart (SB5-SB10).
/// - **Line breaks**: following SB3/SB4, a boundary is placed after every
///   mandatory break (`CR LF`, `LF`, `CR`, `VT`, `FF`, `NEL`, `LS`, `PS`),
///   each such separator forming its own split. There is no paragraph-level
///   segmentation: UAX #29 does not define one.
#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct SentenceSplit;

impl SentenceSplit {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SentenceSplit {
    fn default() -> Self {
        Self::new()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OffsetReferential;
    use crate::OffsetType;

    fn split_str(input: &str) -> Vec<(String, (usize, usize))> {
        let pretok = SentenceSplit::new();
        let mut pretokenized = PreTokenizedString::from(input);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s.to_owned(), o))
            .collect()
    }

    #[test]
    fn basic_multilingual() {
        // English: abbreviations are not split (SB6/SB7/SB8), trailing
        // spaces belong to the preceding sentence (SB11)
        assert_eq!(
            split_str("Mr. Smith went.  He left."),
            vec![
                ("Mr. Smith went.  ".to_string(), (0, 17)),
                ("He left.".to_string(), (17, 25))
            ]
        );
        // Chinese: 。 is a sentence terminator
        assert_eq!(
            split_str("他说：你好。转身走了。"),
            vec![
                ("他说：你好。".to_string(), (0, 18)),
                ("转身走了。".to_string(), (18, 33))
            ]
        );
        // Japanese: quotes stay attached to the sentence they close (SB8a/SB9)
        assert_eq!(
            split_str("彼は言った。「こんにちは。」そして去った。"),
            vec![
                ("彼は言った。".to_string(), (0, 18)),
                ("「こんにちは。」".to_string(), (18, 42)),
                ("そして去った。".to_string(), (42, 63))
            ]
        );
    }

    #[test]
    fn line_breaks() {
        // SB3: CRLF is a single unit; SB4: break after every Sep/CR/LF,
        // each separator becomes its own split (no paragraph level)
        assert_eq!(
            split_str("a\r\nb\n\nc"),
            vec![
                ("a".to_string(), (0, 1)),
                ("\r\n".to_string(), (1, 3)),
                ("b".to_string(), (3, 4)),
                ("\n".to_string(), (4, 5)),
                ("\n".to_string(), (5, 6)),
                ("c".to_string(), (6, 7))
            ]
        );
    }

    #[test]
    fn lossless() {
        // Concatenation of splits must always be identical to the input
        let inputs = [
            "",
            " ",
            ".",
            "\n\n",
            "Mr. Smith went.  He left.",
            "他说：你好。转身走了。",
            "组合 a\u{0301} 与 👨\u{200D}👩\u{200D}👧 emoji。下一句",
            "末尾无终止符",
            "Wait… what?! Really?!…",
            "3.14 is pi. 1.5.2 is not.",
        ];
        for input in inputs {
            let splits = split_str(input);
            assert_eq!(
                splits.iter().map(|(s, _)| s.as_str()).collect::<String>(),
                input,
                "lossless violated for {input:?}"
            );
            // Offsets must cover the whole input contiguously
            let mut expected_start = 0;
            for (_, (start, end)) in &splits {
                assert_eq!(*start, expected_start);
                expected_start = *end;
            }
            assert_eq!(expected_start, input.len());
        }
    }

    #[test]
    fn grapheme_clusters_not_split() {
        // SB5: never break before Extend/Format/ZWJ — combining sequences and
        // ZWJ emoji must stay inside a single split
        let splits = split_str("Cafe\u{0301} 👨\u{200D}👩\u{200D}👧 done.");
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].0, "Cafe\u{0301} 👨\u{200D}👩\u{200D}👧 done.");
    }
}
