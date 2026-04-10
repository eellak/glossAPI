#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScriptMetrics {
    pub non_whitespace_chars: u64,
    pub greek_char_count: u64,
    pub latin_char_count: u64,
    pub greek_word_count: u64,
    pub polytonic_word_count: u64,
}

impl ScriptMetrics {
    #[inline]
    pub fn percentage_greek(&self) -> f64 {
        if self.non_whitespace_chars > 0 {
            100.0 * self.greek_char_count as f64 / self.non_whitespace_chars as f64
        } else {
            0.0
        }
    }

    #[inline]
    pub fn latin_percentage(&self) -> f64 {
        if self.non_whitespace_chars > 0 {
            100.0 * self.latin_char_count as f64 / self.non_whitespace_chars as f64
        } else {
            0.0
        }
    }

    #[inline]
    pub fn polytonic_ratio(&self) -> f64 {
        if self.greek_word_count > 0 {
            self.polytonic_word_count as f64 / self.greek_word_count as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScriptScanner {
    metrics: ScriptMetrics,
    token_has_greek: bool,
    token_has_polytonic: bool,
    in_token: bool,
}

impl ScriptScanner {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn observe_char(&mut self, ch: char) {
        if ch.is_whitespace() {
            self.finish_token();
            return;
        }

        self.in_token = true;
        self.metrics.non_whitespace_chars += 1;

        let cp = ch as u32;
        if is_greek(cp) {
            self.metrics.greek_char_count += 1;
            self.token_has_greek = true;
            if is_polytonic_codepoint(cp) {
                self.token_has_polytonic = true;
            }
        } else if is_ascii_latin(cp) {
            self.metrics.latin_char_count += 1;
        } else if is_combining_mark(cp) {
            self.token_has_polytonic = true;
        }
    }

    #[inline]
    pub fn observe_str(&mut self, text: &str) {
        for ch in text.chars() {
            self.observe_char(ch);
        }
    }

    #[inline]
    pub fn finish_token(&mut self) {
        if !self.in_token {
            return;
        }
        if self.token_has_greek {
            self.metrics.greek_word_count += 1;
            if self.token_has_polytonic {
                self.metrics.polytonic_word_count += 1;
            }
        }
        self.in_token = false;
        self.token_has_greek = false;
        self.token_has_polytonic = false;
    }

    #[inline]
    pub fn finish(mut self) -> ScriptMetrics {
        self.finish_token();
        self.metrics
    }
}

#[inline(always)]
pub fn is_greek(cp: u32) -> bool {
    (0x0370..=0x03FF).contains(&cp) || (0x1F00..=0x1FFF).contains(&cp)
}

#[inline(always)]
pub fn is_combining_mark(cp: u32) -> bool {
    (0x0300..=0x036F).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
}

#[inline(always)]
pub fn is_ascii_latin(cp: u32) -> bool {
    (0x41..=0x5A).contains(&cp) || (0x61..=0x7A).contains(&cp)
}

#[inline(always)]
pub fn is_polytonic_codepoint(cp: u32) -> bool {
    (0x1F00..=0x1FFF).contains(&cp)
}

#[inline]
pub fn scan_script_metrics(text: &str) -> ScriptMetrics {
    let mut scanner = ScriptScanner::new();
    scanner.observe_str(text);
    scanner.finish()
}

#[cfg(test)]
mod tests {
    use super::{scan_script_metrics, ScriptScanner};

    #[test]
    fn scanner_counts_greek_latin_and_polytonic_words() {
        let metrics = scan_script_metrics("Αυτή abc Καὶ");
        assert!(metrics.greek_char_count > 0);
        assert_eq!(metrics.latin_char_count, 3);
        assert_eq!(metrics.greek_word_count, 2);
        assert_eq!(metrics.polytonic_word_count, 1);
        assert!(metrics.percentage_greek() > metrics.latin_percentage());
    }

    #[test]
    fn scanner_flushes_on_line_boundaries() {
        let mut scanner = ScriptScanner::new();
        scanner.observe_str("Καὶ\n");
        scanner.observe_str("αὕτη");
        let metrics = scanner.finish();
        assert_eq!(metrics.greek_word_count, 2);
        assert_eq!(metrics.polytonic_word_count, 2);
    }
}
