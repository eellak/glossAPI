//! A small matcher with Python `re` semantics for the layout v2 patterns (copied verbatim from the reference into
//! `tables.rs`). Supported: literals, `[...]` classes with ranges and escapes, `\s \S \d`, `\uXXXX`, groups,
//! `(?!...)`, alternation, `? * + {m} {m,} {m,n}`, `^` (string start) and `$` (end, or before a final newline).
//! Booleans only: a match exists iff some path through the pattern matches, so the search order does not matter.
//! The pattern is compiled to a small instruction program and searched depth first with an explicit stack and a
//! visited set over (instruction, position) (RE2's "bit state" method): the work is bounded by program size x text
//! length, so nested `\s*` on long whitespace runs (Python's own backtracking is quadratic there) and deep
//! repetitions neither blow up nor grow the call stack.
//!
//! IGNORECASE follows CPython 3.11 `re._compiler` for str patterns: a cased literal `L` matches `ch` when
//! `lower(ch)` is `lower(L)` or one of `_EXTRA_CASES[lower(L)]`; a class with a cased member matches when `lower(ch)`
//! is in the set `{lower(x)} + extra cases` of its literals and ranges (a class with no cased member compares `ch`
//! itself); `\s`, `\d` outside a class never fold. `lower` = `_sre.unicode_tolower`.
use super::pyuni::{is_decimal, is_space, re_extra_cases, re_iscased, re_lower};

#[derive(Debug, Clone)]
enum Item {
    Lit(u32),
    Range(u32, u32),
    Space,
    NotSpace,
    Digit,
}

#[derive(Debug, Clone)]
enum Node {
    /// Literal; with IGNORECASE the accepted lowered set.
    Lit(u32),
    LitFold(Vec<u32>),
    Class { items: Vec<Item>, fold: bool },
    Space,
    NotSpace,
    Digit,
    Start,
    End,
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Rep(Box<Node>, u32, u32),
    NegLook(Box<Node>),
}

pub struct Regex {
    prog: Prog,
}

/// One char test (the single-char nodes).
#[derive(Debug, Clone)]
enum Test {
    Lit(u32),
    LitFold(Vec<u32>),
    Class { items: Vec<Item>, fold: bool },
    Space,
    NotSpace,
    Digit,
}

#[derive(Debug, Clone)]
enum Inst {
    Char(Test),
    Split(usize, usize),
    Jmp(usize),
    Start,
    End,
    /// Negative lookahead: index of the sub-program in `Prog::subs`.
    NegLook(usize),
    Match,
}

#[derive(Debug, Clone, Default)]
struct Prog {
    insts: Vec<Inst>,
    subs: Vec<Prog>,
}

impl Prog {
    fn compile(node: &Node) -> Prog {
        let mut p = Prog::default();
        p.emit(node);
        p.insts.push(Inst::Match);
        p
    }

    fn emit(&mut self, n: &Node) {
        match n {
            Node::Lit(v) => self.insts.push(Inst::Char(Test::Lit(*v))),
            Node::LitFold(set) => self.insts.push(Inst::Char(Test::LitFold(set.clone()))),
            Node::Class { items, fold } => self.insts.push(Inst::Char(Test::Class { items: items.clone(), fold: *fold })),
            Node::Space => self.insts.push(Inst::Char(Test::Space)),
            Node::NotSpace => self.insts.push(Inst::Char(Test::NotSpace)),
            Node::Digit => self.insts.push(Inst::Char(Test::Digit)),
            Node::Start => self.insts.push(Inst::Start),
            Node::End => self.insts.push(Inst::End),
            Node::Seq(items) => items.iter().for_each(|x| self.emit(x)),
            Node::Alt(alts) => {
                let mut jumps = Vec::new();
                for (k, a) in alts.iter().enumerate() {
                    if k + 1 < alts.len() {
                        let split = self.insts.len();
                        self.insts.push(Inst::Split(split + 1, 0));
                        self.emit(a);
                        jumps.push(self.insts.len());
                        self.insts.push(Inst::Jmp(0));
                        let next = self.insts.len();
                        self.insts[split] = Inst::Split(split + 1, next);
                    } else {
                        self.emit(a);
                    }
                }
                let end = self.insts.len();
                for j in jumps {
                    self.insts[j] = Inst::Jmp(end);
                }
            }
            Node::Rep(inner, lo, hi) => {
                for _ in 0..*lo {
                    self.emit(inner);
                }
                if *hi == u32::MAX {
                    // L: Split(body, end); body; Jmp L   (every atom consumes a char, so the loop terminates)
                    let l = self.insts.len();
                    self.insts.push(Inst::Split(l + 1, 0));
                    self.emit(inner);
                    self.insts.push(Inst::Jmp(l));
                    let end = self.insts.len();
                    self.insts[l] = Inst::Split(l + 1, end);
                } else {
                    // (x(x(x)?)?)? for the optional copies
                    let mut splits = Vec::new();
                    for _ in *lo..*hi {
                        splits.push(self.insts.len());
                        self.insts.push(Inst::Split(0, 0));
                        self.emit(inner);
                    }
                    let end = self.insts.len();
                    for s in splits {
                        self.insts[s] = Inst::Split(s + 1, end);
                    }
                }
            }
            Node::NegLook(inner) => {
                self.subs.push(Prog::compile(inner));
                self.insts.push(Inst::NegLook(self.subs.len() - 1));
            }
        }
    }

    /// Is there a path from any of `starts` to `Match` with `accept(end)`? A plain depth-first walk first (most
    /// line-start patterns fail within a few steps); past `BUDGET` steps the walk restarts with the visited set,
    /// which bounds the work by program size x text length.
    fn run(&self, s: &[char], starts: impl Iterator<Item = usize> + Clone, accept: &dyn Fn(usize) -> bool) -> bool {
        const BUDGET: usize = 512;
        match self.walk(s, starts.clone(), accept, Some(BUDGET)) {
            Some(found) => found,
            None => self.walk(s, starts, accept, None).unwrap_or(false),
        }
    }

    /// `budget = Some(k)`: no visited set, give up (None) after k steps; `None`: visited set, always decides.
    fn walk(
        &self,
        s: &[char],
        starts: impl Iterator<Item = usize>,
        accept: &dyn Fn(usize) -> bool,
        budget: Option<usize>,
    ) -> Option<bool> {
        let n1 = s.len() + 1;
        let mut visited = if budget.is_none() { vec![0u64; (self.insts.len() * n1).div_ceil(64)] } else { Vec::new() };
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(16);
        let mut steps = 0usize;
        for st in starts {
            stack.push((0, st));
            while let Some((pc, pos)) = stack.pop() {
                match budget {
                    Some(b) => {
                        steps += 1;
                        if steps > b {
                            return None;
                        }
                    }
                    None => {
                        let bit = pc * n1 + pos;
                        if visited[bit / 64] & (1 << (bit % 64)) != 0 {
                            continue;
                        }
                        visited[bit / 64] |= 1 << (bit % 64);
                    }
                }
                match &self.insts[pc] {
                    Inst::Char(t) => {
                        if pos < s.len() && test(t, s[pos]) {
                            stack.push((pc + 1, pos + 1));
                        }
                    }
                    Inst::Split(a, b) => {
                        stack.push((*b, pos));
                        stack.push((*a, pos));
                    }
                    Inst::Jmp(a) => stack.push((*a, pos)),
                    Inst::Start => {
                        if pos == 0 {
                            stack.push((pc + 1, pos));
                        }
                    }
                    Inst::End => {
                        if pos == s.len() || (pos + 1 == s.len() && s[pos] == '\n') {
                            stack.push((pc + 1, pos));
                        }
                    }
                    Inst::NegLook(k) => {
                        if !self.subs[*k].run(s, std::iter::once(pos), &|_| true) {
                            stack.push((pc + 1, pos));
                        }
                    }
                    Inst::Match => {
                        if accept(pos) {
                            return Some(true);
                        }
                    }
                }
            }
        }
        Some(false)
    }
}

fn test(t: &Test, ch: char) -> bool {
    let c = ch as u32;
    match t {
        Test::Lit(v) => *v == c,
        Test::LitFold(set) => set.contains(&re_lower(c)),
        Test::Class { items, fold } => {
            let c = if *fold { re_lower(c) } else { c };
            items.iter().any(|it| item_ok(it, c))
        }
        Test::Space => is_space(ch),
        Test::NotSpace => !is_space(ch),
        Test::Digit => is_decimal(ch),
    }
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    icase: bool,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }
    fn next(&mut self) -> char {
        let c = self.s[self.i];
        self.i += 1;
        c
    }
    fn alt(&mut self) -> Node {
        let mut alts = vec![self.seq()];
        while self.peek() == Some('|') {
            self.i += 1;
            alts.push(self.seq());
        }
        if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        }
    }
    fn seq(&mut self) -> Node {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom();
            let atom = self.quant(atom);
            items.push(atom);
        }
        Node::Seq(items)
    }
    fn number(&mut self) -> Option<u32> {
        let st = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.i == st {
            None
        } else {
            self.s[st..self.i].iter().collect::<String>().parse().ok()
        }
    }
    fn quant(&mut self, atom: Node) -> Node {
        match self.peek() {
            Some('?') => {
                self.i += 1;
                Node::Rep(Box::new(atom), 0, 1)
            }
            Some('*') => {
                self.i += 1;
                Node::Rep(Box::new(atom), 0, u32::MAX)
            }
            Some('+') => {
                self.i += 1;
                Node::Rep(Box::new(atom), 1, u32::MAX)
            }
            Some('{') => {
                let save = self.i;
                self.i += 1;
                let lo = self.number();
                let (lo, hi) = if self.peek() == Some(',') {
                    self.i += 1;
                    let hi = self.number();
                    (lo.unwrap_or(0), hi.unwrap_or(u32::MAX))
                } else {
                    match lo {
                        Some(v) => (v, v),
                        None => {
                            self.i = save;
                            return atom;
                        }
                    }
                };
                assert_eq!(self.next(), '}', "pattern: bad repeat");
                Node::Rep(Box::new(atom), lo, hi)
            }
            _ => atom,
        }
    }
    fn hex4(&mut self) -> u32 {
        let h: String = self.s[self.i..self.i + 4].iter().collect();
        self.i += 4;
        u32::from_str_radix(&h, 16).expect("pattern: bad \\u escape")
    }
    fn lit(&self, c: u32) -> Node {
        if self.icase && re_iscased(c) {
            let lo = re_lower(c);
            let mut set = vec![lo];
            set.extend_from_slice(re_extra_cases(lo));
            Node::LitFold(set)
        } else {
            Node::Lit(c)
        }
    }
    fn atom(&mut self) -> Node {
        let c = self.next();
        match c {
            '(' => {
                let neg = if self.s[self.i..].starts_with(&['?', '!']) {
                    self.i += 2;
                    true
                } else {
                    false
                };
                let inner = self.alt();
                assert_eq!(self.next(), ')', "pattern: unbalanced group");
                if neg {
                    Node::NegLook(Box::new(inner))
                } else {
                    inner
                }
            }
            '[' => self.class(),
            '^' => Node::Start,
            '$' => Node::End,
            '\\' => {
                let e = self.next();
                match e {
                    's' => Node::Space,
                    'S' => Node::NotSpace,
                    'd' => Node::Digit,
                    'u' => {
                        let v = self.hex4();
                        self.lit(v)
                    }
                    other => {
                        assert!(!other.is_ascii_alphanumeric(), "pattern: unsupported escape \\{other}");
                        self.lit(other as u32)
                    }
                }
            }
            '.' => panic!("pattern: '.' is not supported"),
            other => self.lit(other as u32),
        }
    }
    fn class_item(&mut self) -> Item {
        let c = self.next();
        if c == '\\' {
            let e = self.next();
            return match e {
                's' => Item::Space,
                'S' => Item::NotSpace,
                'd' => Item::Digit,
                'u' => Item::Lit(self.hex4()),
                other => {
                    assert!(!other.is_ascii_alphanumeric(), "pattern: unsupported class escape \\{other}");
                    Item::Lit(other as u32)
                }
            };
        }
        Item::Lit(c as u32)
    }
    fn class(&mut self) -> Node {
        assert_ne!(self.peek(), Some('^'), "pattern: negated classes are not supported");
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let c = self.peek().expect("pattern: unterminated class");
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            let a = self.class_item();
            if self.peek() == Some('-') && self.s.get(self.i + 1) != Some(&']') {
                if let Item::Lit(lo) = a {
                    self.i += 1;
                    match self.class_item() {
                        Item::Lit(hi) => {
                            assert!(lo <= hi, "pattern: bad range");
                            items.push(Item::Range(lo, hi));
                        }
                        _ => panic!("pattern: bad range end"),
                    }
                    continue;
                }
            }
            items.push(a);
        }
        if !self.icase {
            return Node::Class { items, fold: false };
        }
        // CPython `_optimize_charset` with fixup: literals and ranges are lowered (+ extra cases); `hascased` decides
        // whether the input char is lowered at match time.
        let mut hascased = false;
        let mut out = Vec::new();
        for it in items {
            match it {
                Item::Lit(v) => {
                    let lo = re_lower(v);
                    out.push(Item::Lit(lo));
                    for &k in re_extra_cases(lo) {
                        out.push(Item::Lit(k));
                    }
                    hascased |= re_iscased(v);
                }
                Item::Range(a, b) => {
                    for v in a..=b {
                        let lo = re_lower(v);
                        out.push(Item::Lit(lo));
                        for &k in re_extra_cases(lo) {
                            out.push(Item::Lit(k));
                        }
                        hascased |= re_iscased(v);
                    }
                }
                other => out.push(other),
            }
        }
        if hascased {
            Node::Class { items: out, fold: true }
        } else {
            // no cased member: CPython emits a plain IN over the lowered (= unchanged) set
            Node::Class { items: out, fold: false }
        }
    }
}

fn item_ok(it: &Item, c: u32) -> bool {
    match *it {
        Item::Lit(v) => v == c,
        Item::Range(a, b) => a <= c && c <= b,
        Item::Space => char::from_u32(c).is_some_and(is_space),
        Item::NotSpace => !char::from_u32(c).is_some_and(is_space),
        Item::Digit => char::from_u32(c).is_some_and(is_decimal),
    }
}

impl Regex {
    pub fn new(pattern: &str, icase: bool) -> Regex {
        let s: Vec<char> = pattern.chars().collect();
        let mut p = Parser { s: &s, i: 0, icase };
        let node = p.alt();
        assert_eq!(p.i, s.len(), "pattern: trailing input");
        Regex { prog: Prog::compile(&node) }
    }

    /// `pattern.match(s)` is not None.
    pub fn is_match(&self, s: &[char]) -> bool {
        self.prog.run(s, std::iter::once(0), &|_| true)
    }

    /// `pattern.fullmatch(s)` is not None.
    pub fn is_fullmatch(&self, s: &[char]) -> bool {
        let n = s.len();
        self.prog.run(s, std::iter::once(0), &|p| p == n)
    }

    /// `pattern.search(s)` is not None.
    pub fn is_search(&self, s: &[char]) -> bool {
        self.prog.run(s, 0..=s.len(), &|_| true)
    }
}
