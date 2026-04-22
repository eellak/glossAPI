#!/usr/bin/env bash
# Overnight orchestrator — runs all four tokenizer trainings sequentially
# on the newly-cleaned corpus shards. Starts by cleaning HPLT (worker-side
# combined driver) then launches each BPE train.
#
# Precondition: /home/foivos/runs/raw_clean_stats_20260422/shards/ already
# contains the glossapi-only cleaned shards (stage 1 done).
#
# Run: nohup ./orchestrate_all_4_tokenizers.sh > orchestrate.log 2>&1 &
set -euo pipefail

RUN_ROOT=/home/foivos/runs
HPLT_STATS_DIR=$RUN_ROOT/hplt_clean_stats_20260422
GLOSSAPI_SHARDS=$RUN_ROOT/raw_clean_stats_20260422/shards
HPLT_SHARDS=$HPLT_STATS_DIR/shards
CATEGORY_SPECS=/home/foivos/token_noise_review_workspace/corpus_clean_normalization/specs/three_counter_spec_20260421.json
THRESHOLDS=/home/foivos/runs/thresholds_v2_calibrated.json
BASE_SNAPSHOT_DIR=/home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415

source /home/foivos/venvs/glossapi-corpus-clean/bin/activate

log() { echo "[$(date -Is)] $*"; }

### STAGE 2 — Clean HPLT portion (from v2; we only add new per-line + drops)
if [ ! -d "$HPLT_SHARDS" ] || [ $(ls "$HPLT_SHARDS"/*.txt.gz 2>/dev/null | wc -l) -lt 200 ]; then
  log "STAGE 2 — cleaning HPLT"
  mkdir -p "$HPLT_STATS_DIR/stats" "$HPLT_SHARDS"
  python3 /home/foivos/glossAPI-development/cleaning_scripts/clean_and_stats_full.py \
    --input-glob '/home/foivos/data/glossapi_work/hf_release_publish_cleaned_v2/data/HPLT*.parquet' \
    --stats-dir "$HPLT_STATS_DIR/stats" \
    --text-shards-dir "$HPLT_SHARDS" \
    --category-specs "$CATEGORY_SPECS" \
    --thresholds "$THRESHOLDS" \
    --workers 48
  log "STAGE 2 — HPLT clean done"
fi

### STAGE 3 — Train 4 tokenizers sequentially

log "STAGE 3.1 — fresh glossapi-only"
python3 /home/foivos/glossAPI-development/cleaning_scripts/train_bpe_from_text_shards.py \
  --mode fresh \
  --shards-dir "$GLOSSAPI_SHARDS" \
  --vocab-size 50000 \
  --output-dir "$RUN_ROOT/bpe_fresh_glossapi_only_cleaned"
log "3.1 done"

log "STAGE 3.2 — fresh glossapi+hplt 70/30"
python3 /home/foivos/glossAPI-development/cleaning_scripts/train_bpe_from_text_shards.py \
  --mode fresh \
  --shards-dir "$GLOSSAPI_SHARDS" \
  --hplt-shards-dir "$HPLT_SHARDS" \
  --hplt-ratio 0.3 \
  --vocab-size 50000 \
  --output-dir "$RUN_ROOT/bpe_fresh_glossapi_plus_hplt_70_30_cleaned"
log "3.2 done"

log "STAGE 3.3 — continuous glossapi-only (target 156672)"
python3 /home/foivos/glossAPI-development/cleaning_scripts/train_bpe_from_text_shards.py \
  --mode continuous \
  --shards-dir "$GLOSSAPI_SHARDS" \
  --target-vocab-size 156672 \
  --base-tokenizer-dir "$BASE_SNAPSHOT_DIR" \
  --output-dir "$RUN_ROOT/bpe_continuous_glossapi_only_cleaned"
log "3.3 done"

log "STAGE 3.4 — continuous glossapi+hplt 70/30 (target 156672)"
python3 /home/foivos/glossAPI-development/cleaning_scripts/train_bpe_from_text_shards.py \
  --mode continuous \
  --shards-dir "$GLOSSAPI_SHARDS" \
  --hplt-shards-dir "$HPLT_SHARDS" \
  --hplt-ratio 0.3 \
  --target-vocab-size 156672 \
  --base-tokenizer-dir "$BASE_SNAPSHOT_DIR" \
  --output-dir "$RUN_ROOT/bpe_continuous_glossapi_plus_hplt_70_30_cleaned"
log "3.4 done"

### STAGE 4 — Per-tokenizer id_map + quick stats

log "STAGE 4 — generating id_maps"
for d in "$RUN_ROOT"/bpe_*_cleaned; do
  name=$(basename "$d")
  if [ -f "$d/tokenizer.json" ]; then
    python3 - <<PY
import json
from pathlib import Path
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("$d", use_fast=True)
    specials = set(tok.all_special_ids) | set(tok.added_tokens_decoder.keys())
    vocab = tok.get_vocab()
    m = {}
    for s, i in vocab.items():
        if i in specials: continue
        m[str(i)] = tok.decode([i], clean_up_tokenization_spaces=False)
    m = dict(sorted(m.items(), key=lambda kv: int(kv[0])))
    out = Path("$d") / "id_map.json"
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out}: {len(m)} entries")
except Exception as exc:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("$d/tokenizer.json")
    vocab = tok.get_vocab()
    m = {str(v): k for k, v in vocab.items()}
    m = dict(sorted(m.items(), key=lambda kv: int(kv[0])))
    Path("$d/id_map.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"(fallback) {len(m)} entries")
PY
  fi
done

log "ALL 4 TRAININGS COMPLETE"
ls -la "$RUN_ROOT"/bpe_*_cleaned/
