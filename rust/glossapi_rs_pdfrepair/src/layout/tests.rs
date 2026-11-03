//! Unit tests of the layout v2 port. Expected values were computed with the reference interpreter (Python 3.11.3);
//! the full differential (real pages, seeded fuzz, every code point) runs from Python (tests/test_layout_v2_rust.py).
use super::k5chars::{dump, parse, CharTable};
use super::pyre::Regex;
use super::pyuni::{casefold, nfc, round2};
use super::*;

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

#[test]
fn tables_come_from_the_reference_interpreter() {
    assert_eq!(tables::PYTHON_VERSION, "3.11.3");
    assert_eq!(tables::UNIDATA_VERSION, "14.0.0");
}

#[test]
fn python_whitespace_and_digits() {
    for c in ['\u{1c}', '\u{1f}', '\u{85}', '\u{a0}', '\u{3000}', '\t', '\n'] {
        assert!(is_space(c), "{c:?}");
    }
    assert!(!is_space('\u{200b}') && !is_space('\u{180e}'));
    assert!(is_digit('²') && !is_decimal('²'));
    assert!(is_decimal('٣') && pyuni::is_numeric('½') && !is_digit('½'));
}

#[test]
fn re_ignorecase_follows_sre() {
    let page = Regex::new(tables::PAGE_NUM.0, tables::PAGE_NUM.1);
    for s in ["12", "- 7 -", "xiv", "XIV", "İ", "ιβ΄", "ΙΒ΄", "Ωʹ", "ϐ'", "\u{1fbe}΄"] {
        assert!(page.is_match(&chars(s)), "{s}");
    }
    for s in ["12345", "abc", "ιβγδ΄", "1 2"] {
        assert!(!page.is_match(&chars(s)), "{s}");
    }
    let bold = Regex::new(tables::BOLD_NAME.0, tables::BOLD_NAME.1);
    assert!(bold.is_search(&chars("ABCD+Times-Bold")));
    assert!(bold.is_search(&chars("Font-B\n")), "$ matches before a final newline");
    assert!(bold.is_search(&chars("ſemiBOLD")) && bold.is_search(&chars("BX9")));
    assert!(!bold.is_search(&chars("Times-Roman")) && !bold.is_search(&chars("xgrx")));
}

#[test]
fn deep_repetition_does_not_grow_the_stack() {
    let bullet = Regex::new(tables::BULLET.0, tables::BULLET.1);
    let mut s: Vec<char> = vec![' '; 500_000];
    s.push('•');
    s.push(' ');
    assert!(bullet.is_match(&s));
    let numbered = Regex::new(tables::NUMBERED.0, tables::NUMBERED.1);
    let mut s: Vec<char> = "1".chars().collect();
    for _ in 0..200_000 {
        s.extend(".12".chars());
    }
    s.push(' ');
    assert!(numbered.is_match(&s));
}

#[test]
fn difflib_matching_blocks() {
    let m = |a: &str, b: &str| {
        let (a, b) = (chars(a), chars(b));
        difflib::Matcher::new(&a, &b).matching_blocks()
    };
    assert_eq!(m("abxcd", "abcd"), vec![(0, 0, 2), (3, 2, 2)]);
    assert_eq!(m("qabxcabc", "abcabd"), vec![(1, 0, 2), (4, 2, 3)]);
    assert_eq!(m("", "abc"), vec![]);
    assert_eq!(m("aaaa", "aa"), vec![(0, 0, 2)]);
    assert_eq!(m("Σελίδα", "Σελίδαα"), vec![(0, 0, 6)]);
}

#[test]
fn python_rounding_and_unicode() {
    for (x, want) in [(0.125, 0.12), (0.375, 0.38), (2.675, 2.67), (1.005, 1.0), (-0.125, -0.12), (72.125, 72.12)] {
        assert_eq!(round2(x), want);
    }
    assert_eq!(nfc(&chars("e\u{301}\u{1100}\u{1161}\u{11a8}")), chars("é각"));
    assert_eq!(casefold(&chars("ßΣİ")), chars("ssσi\u{307}"));
    assert_eq!(norm_sig(&chars("  Σελίδα\u{a0} 12 ")), chars("σελίδα ##"));
}

#[test]
fn utf8_replacement_matches_python() {
    for (bytes, want) in [
        (&b"ab\xffc"[..], "ab\u{fffd}c"),
        (b"\xe2\x82", "\u{fffd}"),
        (b"\xe2\x82x", "\u{fffd}x"),
        (b"\xf0\x9f\x98", "\u{fffd}"),
        (b"\xed\xa0\x80", "\u{fffd}\u{fffd}\u{fffd}"),
        (b"\xc0\xaf", "\u{fffd}\u{fffd}"),
        (b"\xf4\x90\x80\x80", "\u{fffd}\u{fffd}\u{fffd}\u{fffd}"),
        (b"a\x00b", "a\u{0}b"),
    ] {
        assert_eq!(String::from_utf8_lossy(bytes), want);
    }
}

fn row(t: &mut CharTable, text: &str, x: f64, top: f64, size: f64, font: u32) {
    let mut cx = x;
    for c in text.chars() {
        let w = 0.5 * size;
        let b = [cx, t.page_h - top - size, cx + w, t.page_h - top];
        t.ch.push(c as u32);
        t.bbox.push(b);
        t.loose.push(b);
        t.size.push(size as f32);
        t.weight.push(400);
        t.angle.push(0.0);
        t.font.push(font);
        t.flags.push(0);
        cx += w;
    }
}

fn newline(t: &mut CharTable) {
    for c in ['\r', '\n'] {
        t.ch.push(c as u32);
        t.bbox.push([0.0; 4]);
        t.loose.push([0.0; 4]);
        t.size.push(0.0);
        t.weight.push(0);
        t.angle.push(0.0);
        t.font.push(0);
        t.flags.push(GENERATED);
    }
}

fn sample() -> CharTable {
    let mut t = CharTable { page_w: 595.0, page_h: 842.0, page: 5, fonts: vec!["Times".into(), "Times-Bold".into()], ..Default::default() };
    row(&mut t, "1. Εισαγωγή στο θέμα", 72.0, 70.0, 16.0, 1);
    newline(&mut t);
    let para = [
        "Αυτή είναι μια γραμμή κειμένου που φτάνει μέχρι το δεξί περιθώριο της σελίδας εδώ.",
        "και συνεχίζει στη δεύτερη γραμμή της ίδιας παραγράφου με το ίδιο μέγεθος και ύφος.",
        "Τελευταία γραμμή.",
    ];
    for (i, s) in para.iter().enumerate() {
        row(&mut t, s, 72.0, 100.0 + 12.0 * i as f64, 10.0, 0);
        newline(&mut t);
    }
    row(&mut t, "57", 290.0, 800.0, 10.0, 0);
    t
}

#[test]
fn label_page_keeps_the_text_and_labels_a_heading() {
    let t = sample();
    let plain: String = fixture_text(&t).into_iter().collect();
    let g = page_geo(&chars(&plain), &t).unwrap();
    let ctx = document_context(&[(5, g)]).unwrap();
    let res = label_page(&plain, &t, &ctx, 5).unwrap();
    let labels: Vec<&str> = res.blocks.iter().map(|b| b.label).collect();
    assert_eq!(labels, vec!["section_header", "paragraph", "page_footer"]);
    let pc = chars(&plain);
    let joined: Vec<String> = res.blocks.iter().map(|b| pc[b.start..b.end].iter().collect()).collect();
    assert_eq!(joined.join("\n"), plain);
    assert!(page_markdown(&plain, &res.blocks).starts_with("## 1. Εισαγωγή"));
}

#[test]
fn k5chars_round_trip_and_bad_input() {
    let t = sample();
    let sha = [7u8; 32];
    let data = dump(&t, &sha);
    let back = parse(&data).unwrap();
    assert_eq!(dump(&back, &sha), data);
    for bad in [&data[..10], &data[..data.len() - 1], b"NOTK5CHARS-at-all-but-long-enough-to-read-a-header-xxxxxxxxxxxxxxxxxxxxxxxxx"] {
        assert!(parse(bad).is_err());
    }
    let mut more = data.clone();
    more.push(0);
    assert!(parse(&more).is_err(), "trailing bytes");
    let mut huge = data.clone();
    huge[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(parse(&huge).is_err(), "impossible char count is refused before allocating");
}

#[test]
fn degenerate_pages_do_not_panic() {
    let ctx = DocContext { body: 0.0, heading_sizes: vec![], furniture: vec![], pages: 0 };
    let empty = CharTable { page_w: 595.0, page_h: 842.0, ..Default::default() };
    assert!(label_page("", &empty, &ctx, 1).unwrap().blocks.is_empty());
    assert_eq!(label_page("κείμενο\nχωρίς πίνακα", &empty, &ctx, 1).unwrap().blocks.len(), 1);
    let mut nan = sample();
    nan.angle[3] = f32::NAN;
    assert!(label_page("x", &nan, &ctx, 1).is_err(), "the reference raises on a NaN angle");
    let mut inf = sample();
    inf.page_w = f64::INFINITY;
    assert!(label_page("x", &inf, &ctx, 1).is_err());
}
