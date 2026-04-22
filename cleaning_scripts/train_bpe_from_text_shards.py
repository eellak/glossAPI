"""Train a fresh or continuous BPE tokenizer from gzipped text shards.

Handles both modes and the 70/30 glossapi+HPLT mix:

  - Fresh mode: calls `base.train_new_from_iterator(it, vocab_size=VS)`
    where base = AutoTokenizer(swiss-ai/Apertus-8B-2509).
  - Continuous mode: loads the Apertus backend_tokenizer.runtime.json,
    trains additional merges via `tokenizer.train_from_iterator(...)`
    to reach target_vocab_size (Apertus 131,072 + 25,600 new = 156,672).

Input is one or more `--shards-dir` directories of *.txt.gz files (one
doc per line, no internal newlines — what clean_and_stats_full.py
produces). Supports `--hplt-shards-dir` + `--hplt-ratio` to interleave
a subsample of HPLT lines into the glossapi-only stream at the
requested fraction.

Usage:
  # Fresh glossapi-only
  python3 train_bpe_from_text_shards.py \\
    --shards-dir /home/foivos/runs/raw_clean_stats_20260422/shards \\
    --mode fresh --vocab-size 50000 \\
    --output-dir /home/foivos/runs/bpe_fresh_glossapi_only_cleaned

  # Fresh glossapi+hplt 70/30
  python3 train_bpe_from_text_shards.py \\
    --shards-dir /home/foivos/runs/raw_clean_stats_20260422/shards \\
    --hplt-shards-dir /home/foivos/runs/hplt_clean_stats_20260422/shards \\
    --hplt-ratio 0.3 \\
    --mode fresh --vocab-size 50000 \\
    --output-dir /home/foivos/runs/bpe_fresh_glossapi_plus_hplt_70_30_cleaned

  # Continuous glossapi-only (156672)
  python3 train_bpe_from_text_shards.py \\
    --shards-dir /home/foivos/runs/raw_clean_stats_20260422/shards \\
    --mode continuous --target-vocab-size 156672 \\
    --base-tokenizer-dir /home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415 \\
    --output-dir /home/foivos/runs/bpe_continuous_glossapi_only_cleaned
"""
from __future__ import annotations

import argparse
import glob as globmod
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional


def iter_shards(
    glossapi_dir: Optional[Path],
    hplt_dir: Optional[Path],
    hplt_ratio: float,
    seed: int,
    chunk: int,
) -> Iterator[List[str]]:
    """Yield chunks of lines. Interleaves HPLT at `hplt_ratio` if provided.

    hplt_ratio 0.3 means 30% of yielded lines come from HPLT, 70% from
    glossapi. Implemented by random per-line choice with Bernoulli(0.3)
    between the two streams (both streams are iterated concurrently via
    independent generators).
    """
    rng = random.Random(seed)

    def _iter_dir(p: Path) -> Iterator[str]:
        for f in sorted(p.glob("*.txt.gz")):
            with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        yield line

    glossapi_iter = _iter_dir(glossapi_dir) if glossapi_dir else iter(())
    hplt_iter = _iter_dir(hplt_dir) if hplt_dir else iter(())

    buf: List[str] = []
    glossapi_exhausted = hplt_dir is None
    hplt_exhausted = hplt_dir is None
    total = 0
    start = time.time()
    while True:
        use_hplt = (hplt_dir is not None and not hplt_exhausted and
                    (rng.random() < hplt_ratio or glossapi_exhausted))
        try:
            if use_hplt:
                line = next(hplt_iter)
            else:
                line = next(glossapi_iter)
        except StopIteration:
            if use_hplt:
                hplt_exhausted = True
            else:
                glossapi_exhausted = True
            if glossapi_exhausted and hplt_exhausted:
                break
            continue

        buf.append(line)
        total += 1
        if len(buf) >= chunk:
            yield buf
            buf = []
            if total % 500_000 == 0:
                el = time.time() - start
                rate = total / el if el else 0
                print(f"[iter] {total} lines yielded, elapsed={el:.0f}s rate={rate:.0f}/s", flush=True)
    if buf:
        yield buf


def fresh_train(
    shards_dir: Path,
    hplt_shards_dir: Optional[Path],
    hplt_ratio: float,
    vocab_size: int,
    output_dir: Path,
    base_tokenizer: str,
    seed: int,
    chunk: int,
) -> None:
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(base_tokenizer, use_fast=True)
    assert getattr(base, "is_fast", False)
    print(f"[fresh] vocab_size={vocab_size} base={base_tokenizer}")
    it = iter_shards(shards_dir, hplt_shards_dir, hplt_ratio, seed, chunk)
    start = time.time()
    trained = base.train_new_from_iterator(it, vocab_size=vocab_size)
    trained.save_pretrained(output_dir)
    print(f"[fresh-done] elapsed={time.time() - start:.0f}s output={output_dir}")


def continuous_train(
    shards_dir: Path,
    hplt_shards_dir: Optional[Path],
    hplt_ratio: float,
    target_vocab_size: int,
    output_dir: Path,
    base_tokenizer_dir: Path,
    seed: int,
    chunk: int,
) -> None:
    from tokenizers import Tokenizer

    runtime_json = Path(base_tokenizer_dir) / "backend_tokenizer.runtime.json"
    if not runtime_json.exists():
        raise FileNotFoundError(f"{runtime_json} not found")
    tok = Tokenizer.from_file(str(runtime_json))
    current = tok.get_vocab_size()
    to_add = target_vocab_size - current
    print(f"[cont] current={current} target={target_vocab_size} adding={to_add}")
    assert to_add > 0, "target_vocab_size must exceed current"

    it = iter_shards(shards_dir, hplt_shards_dir, hplt_ratio, seed, chunk)
    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(
        vocab_size=target_vocab_size,
        initial_alphabet=[],  # keep existing base alphabet
        special_tokens=[],
        show_progress=True,
    )
    start = time.time()
    tok.train_from_iterator(it, trainer=trainer)
    # Save as an HF tokenizer snapshot
    output_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_dir / "tokenizer.json"))
    print(f"[cont-done] elapsed={time.time() - start:.0f}s output={output_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fresh", "continuous"], required=True)
    parser.add_argument("--shards-dir", required=True, type=Path,
                        help="Directory of glossapi cleaned txt.gz shards")
    parser.add_argument("--hplt-shards-dir", type=Path,
                        help="Optional HPLT shards dir for mix mode")
    parser.add_argument("--hplt-ratio", type=float, default=0.0,
                        help="Fraction of yielded lines from HPLT (0.3 for 70/30 mix)")
    parser.add_argument("--vocab-size", type=int, default=50000,
                        help="Fresh mode: target vocab")
    parser.add_argument("--target-vocab-size", type=int, default=156672,
                        help="Continuous mode: target vocab including base")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-tokenizer", default="swiss-ai/Apertus-8B-2509",
                        help="Fresh mode base tokenizer")
    parser.add_argument("--base-tokenizer-dir", type=Path,
                        help="Continuous mode base tokenizer dir (with backend_tokenizer.runtime.json)")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fresh":
        fresh_train(
            args.shards_dir, args.hplt_shards_dir, args.hplt_ratio,
            args.vocab_size, args.output_dir, args.base_tokenizer,
            args.seed, args.chunk,
        )
    else:
        if not args.base_tokenizer_dir:
            print("--base-tokenizer-dir required for continuous mode", file=sys.stderr)
            return 2
        continuous_train(
            args.shards_dir, args.hplt_shards_dir, args.hplt_ratio,
            args.target_vocab_size, args.output_dir, args.base_tokenizer_dir,
            args.seed, args.chunk,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
