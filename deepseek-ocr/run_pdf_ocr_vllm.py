#!/usr/bin/env python3
"""High-throughput DeepSeek-OCR runner backed by vLLM with grounded/clean modes."""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import logging
import re
import time
import html
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# Prefer DeepSeek OCR-specific components if available; otherwise fall back to VL2 equivalents
try:  # vLLM builds without deepseek_ocr module on some releases
    from vllm.model_executor.models.deepseek_ocr import (  # type: ignore
        NGramPerReqLogitsProcessor,
    )
    _DEEPSEEK_OCR_AVAILABLE = True
except Exception:  # pragma: no cover - import-time capability detection
    NGramPerReqLogitsProcessor = None  # type: ignore
    _DEEPSEEK_OCR_AVAILABLE = False

DEFAULT_GROUNDED_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_CLEAN_PROMPT = "<image>\nConvert the document to markdown."
DEFAULT_ROI_PROMPT = (
    "<image>\nYou are given a cropped region from a document. "
    "Transcribe only the content of this region in markdown/HTML. "
    "Leave empty table cells blank (<td></td>) and keep Greek diacritics."
)
DEFAULT_RETRY_LABELS = ("table",)
BASE_VISION_CONFIG = (1024, 640, True)
LARGE_VISION_CONFIG = (1280, 1280, False)
CHECKPOINT_DIR = Path(__file__).resolve().parent / "DeepSeek-OCR"

FULL_WIDTH_BAR = "\uFF5C"
FULL_WIDTH_LT = "\uFF1C"
ORPHAN_META_FRAGMENT_PATTERN = re.compile(
    rf"<[|{FULL_WIDTH_BAR}][^>]{0,64}[|{FULL_WIDTH_BAR}]>", re.DOTALL
)
LEFTOVER_META_PATTERN = re.compile(rf"(?:<|{FULL_WIDTH_LT})(?:\||{FULL_WIDTH_BAR})")
REFDET_BLOCK_PATTERN = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL | re.IGNORECASE
)
REFDET_EXTRACT_PATTERN = re.compile(
    r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL | re.IGNORECASE
)
BOUNDING_BOX_PREFIX_PATTERN = re.compile(r"^\s*[A-Za-z_]+\[\[.*?\]\]\s*")
PLACEHOLDER_CELL_PATTERN = re.compile(
    r"(<td\b[^>]*>)(.*?)(</td>)", re.IGNORECASE | re.DOTALL
)
EMPTY_CELL_PATTERN = re.compile(r"<td\b[^>]*>\s*</td>", re.IGNORECASE | re.DOTALL)
TABLE_BLOCK_PATTERN = re.compile(
    r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL
)
NBSP_PATTERN = re.compile(r"(?:&nbsp;|\u00A0)+", re.IGNORECASE)
PLACEHOLDER_VALUES = {
    "none",
    "n/a",
    "na",
    "null",
    "--",
    "—",
    "–",
    "−",
    "-",
    "•",
    "·",
    "[[blank page]]",
    "[blank]",
    "(blank)",
}
PLACEHOLDER_LOGIT_BIAS_VALUE = -1.2
PLACEHOLDER_BIAS_STRINGS = (
    " None",
    " none",
    "None",
    "none",
    " N/A",
    " n/a",
    "N/A",
    "n/a",
    " [[Blank page]]",
    "[[Blank page]]",
)
DEHYPHEN_PATTERN = re.compile(r"(?<=\w)-\n(?=[a-zα-ωά-ώ])", re.IGNORECASE)
TIKZ_BLOCK_PATTERN = re.compile(
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.DOTALL
)
LATEX_ARRAY_SPAM_PATTERN = re.compile(
    r"(?:\[\s*\\begin\{array\}.*?\\end\{array\}\s*\]){3,}", re.DOTALL
)
LATEX_DRAW_SPAM_PATTERN = re.compile(
    r"(?:\\draw\s*\([^;]*\);\s*){10,}", re.DOTALL
)
INLINE_LATEX_PATTERN = re.compile(r"\\\((.+?)\\\)")
BLOCK_LATEX_PATTERN = re.compile(r"\\\[\s*(.+?)\s*\\\]", re.DOTALL)
SIMPLE_SUP_PATTERN = re.compile(r"\$\^\{?([A-Za-z0-9+\-]+)\}?\$")
SIMPLE_SUB_PATTERN = re.compile(r"\$_\{?([A-Za-z0-9+\-]+)\}?\$")
CITATION_SUP_PATTERN = re.compile(r"<sup>(\d{2,}(?:[A-Za-z]{1,2})?)</sup>")


@dataclass
class PageJob:
    pdf_path: Path
    page_index: int
    image: Image.Image
    is_blank: bool = False


@dataclass
class ROIJob:
    pdf_path: Path
    page_index: int
    region_index: int
    label: str
    bbox: Tuple[int, int, int, int]
    image: Image.Image


def is_mostly_blank_pix(
    pix: fitz.Pixmap, *, tolerance: int = 8, max_fraction: float = 0.0015
) -> bool:
    buf = pix.samples
    if not buf:
        return True
    channels = 4 if pix.alpha else 3
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size == 0:
        return True
    arr = arr.reshape(-1, channels)
    if channels == 4:
        arr = arr[:, :3]
    if arr.size == 0:
        return True
    if arr.shape[0] > 65536:
        samples = arr[::64]
    else:
        samples = arr
    samples16 = samples.astype(np.int16, copy=False)
    base = samples16[0]
    diff = np.abs(samples16 - base)
    if diff.max() <= tolerance:
        return True
    mask = np.any(diff > tolerance, axis=1)
    return mask.mean() <= max_fraction


def batched(iterable: Iterable[PageJob], size: int) -> Iterator[List[PageJob]]:
    batch: List[PageJob] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def render_page(pdf_path: Path, page_index: int, dpi: int) -> PageJob:
    with fitz.open(pdf_path) as doc:
        if page_index >= doc.page_count:
            raise IndexError(f"Page {page_index} out of bounds for {pdf_path.name}")
        page = doc[page_index]
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        blank = is_mostly_blank_pix(pix)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")
        return PageJob(
            pdf_path=pdf_path,
            page_index=page_index,
            image=image,
            is_blank=blank,
        )


def render_pdf(
    pdf_path: Path,
    dpi: int,
    executor: ThreadPoolExecutor,
    max_pages: Optional[int] = None,
) -> List[PageJob]:
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
    page_count = total_pages if max_pages is None else min(total_pages, max_pages)
    futures = executor.map(
        lambda idx: render_page(pdf_path, idx, dpi), range(page_count)
    )
    jobs = list(futures)
    jobs.sort(key=lambda job: job.page_index)
    return jobs


def ensure_cuda_visible() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not detected; the L4 GPU is required for this script.")


def build_llm(args: argparse.Namespace) -> LLM:
    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Only pass OCR-specific logits processor when present in this vLLM build
        logits_processors=(
            [NGramPerReqLogitsProcessor]
            if NGramPerReqLogitsProcessor is not None
            else []
        ),
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if not args.no_fp8_kv:
        llm_kwargs["kv_cache_dtype"] = "fp8_e4m3"
    if args.enable_fp8_weights:
        llm_kwargs["quantization"] = "fp8"
    if args.mm_encoder_tp_mode and args.mm_encoder_tp_mode != "auto":
        llm_kwargs["mm_encoder_tp_mode"] = args.mm_encoder_tp_mode
    return LLM(**llm_kwargs)


DEFAULT_STOP_SEQUENCES = ("<|", "＜|", "<｜", "｜>")


def build_sampling_params(
    args: argparse.Namespace, mode: str, whitelist_token_ids: set[int]
) -> SamplingParams:
    extra_args = dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids=list(whitelist_token_ids),
    )
    params_kwargs = dict(
        temperature=0.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
        stop=list(DEFAULT_STOP_SEQUENCES),
        extra_args=extra_args,
    )
    return SamplingParams(**params_kwargs)


def strip_prompt_echo(text: str, prompt: str) -> str:
    lines = [
        line.strip()
        for line in prompt.splitlines()
        if line.strip() and line.strip() != "<image>"
    ]
    for line in lines:
        escaped = re.escape(line)
        pattern = re.compile(rf"(?:{escaped})(?:\s+|$)")
        while True:
            new_text = pattern.sub("", text, count=1)
            if new_text == text:
                break
            text = new_text.strip()
    return text


def clean_output(
    text: str, *, keep_refdet: bool, metrics: Optional[dict[str, int]] = None
) -> str:
    text = text.replace("<s>", "").replace("</s>", "")
    if not keep_refdet:
        text = REFDET_BLOCK_PATTERN.sub("", text)
    pattern = get_special_token_pattern(keep_refdet)
    text = pattern.sub("", text)
    text = ORPHAN_META_FRAGMENT_PATTERN.sub("", text)
    text = LATEX_DRAW_SPAM_PATTERN.sub("", text)
    text = TIKZ_BLOCK_PATTERN.sub(
        "[[Figure omitted; refer to original page image]]", text
    )
    text = LATEX_ARRAY_SPAM_PATTERN.sub(
        "[[Matrix omitted; refer to original page image]]", text
    )

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = BOUNDING_BOX_PREFIX_PATTERN.sub("", raw_line)
        line = line.replace("<center>", "").replace("</center>", "")
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] == "":
                continue
            lines.append("")
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines).strip()
    cleaned = INLINE_LATEX_PATTERN.sub(lambda m: f"${m.group(1)}$", cleaned)
    cleaned = BLOCK_LATEX_PATTERN.sub(lambda m: f"$${m.group(1).strip()}$$", cleaned)
    cleaned = SIMPLE_SUP_PATTERN.sub(r"<sup>\1</sup>", cleaned)
    cleaned = SIMPLE_SUB_PATTERN.sub(r"<sub>\1</sub>", cleaned)
    cleaned = prune_placeholder_cells(cleaned, metrics)
    cleaned = drop_empty_tables(cleaned, metrics)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    if CHECKPOINT_DIR.exists():
        model_id = CHECKPOINT_DIR
    else:
        model_id = (
            "deepseek-ai/DeepSeek-OCR"
            if _DEEPSEEK_OCR_AVAILABLE
            else "deepseek-ai/DeepSeek-VL2"
        )
    return AutoTokenizer.from_pretrained(str(model_id), trust_remote_code=True)


def _collect_special_strings(value, bucket: set[str]) -> None:
    if isinstance(value, str):
        if value:
            bucket.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_special_strings(item, bucket)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_special_strings(item, bucket)


@lru_cache(maxsize=2)
def get_special_token_pattern(keep_refdet: bool) -> re.Pattern:
    tokenizer = get_tokenizer()
    specials: set[str] = set()
    _collect_special_strings(tokenizer.all_special_tokens, specials)
    _collect_special_strings(tokenizer.special_tokens_map_extended, specials)
    specials.update({"<|User|>", "<|Assistant|>", "<|grounding|>", "<image>"})
    if keep_refdet:
        specials.difference_update({"<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>"})
    specials = {token for token in specials if token}
    if not specials:
        return re.compile(r"$^")
    variants: set[str] = set()
    for token in specials:
        variants.add(re.escape(token))
        if "|" in token:
            variants.add(re.escape(token.replace("|", FULL_WIDTH_BAR)))
    pattern = "|".join(sorted(variants, key=len, reverse=True))
    return re.compile(f"(?:{pattern})")


@lru_cache(maxsize=1)
def compute_whitelist_token_ids(additional_tags: Tuple[str, ...] = ()) -> set[int]:
    tokenizer = get_tokenizer()
    tags = {
        "<table>",
        "</table>",
        "<tr>",
        "</tr>",
        "<th>",
        "</th>",
        "<td>",
        "</td>",
    }
    tags.update(additional_tags)
    token_ids: set[int] = set()
    for tag in tags:
        encoded = tokenizer.encode(tag, add_special_tokens=False)
        token_ids.update(encoded)
    return token_ids


@lru_cache(maxsize=1)
def get_placeholder_logit_bias() -> dict[int, float]:
    tokenizer = get_tokenizer()
    bias: dict[int, float] = {}
    for candidate in PLACEHOLDER_BIAS_STRINGS:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            bias[ids[0]] = PLACEHOLDER_LOGIT_BIAS_VALUE
    return bias


def _normalize_cell_text(fragment: str) -> str:
    text = html.unescape(fragment)
    text = NBSP_PATTERN.sub(" ", text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_placeholder_content(fragment: str) -> bool:
    normalized = _normalize_cell_text(fragment)
    if not normalized:
        return True
    lowered = normalized.lower()
    compact = re.sub(r"[\s._\-\\/]+", "", lowered)
    return lowered in PLACEHOLDER_VALUES or compact in PLACEHOLDER_VALUES


def prune_placeholder_cells(
    html_text: str, metrics: Optional[dict[str, int]] = None
) -> str:
    def replacer(match: re.Match[str]) -> str:
        opening, body, closing = match.groups()
        if _is_placeholder_content(body):
            if metrics is not None:
                metrics["placeholder_cells_pruned"] = (
                    metrics.get("placeholder_cells_pruned", 0) + 1
                )
            return f"{opening}{closing}"
        return f"{opening}{body}{closing}"

    return PLACEHOLDER_CELL_PATTERN.sub(replacer, html_text)


def drop_empty_tables(
    html_text: str, metrics: Optional[dict[str, int]] = None
) -> str:
    def replace_table(match: re.Match[str]) -> str:
        table_html = match.group(0)
        has_data = False
        for cell_match in PLACEHOLDER_CELL_PATTERN.finditer(table_html):
            cell_body = cell_match.group(2)
            if not _is_placeholder_content(cell_body):
                has_data = True
                break
        if has_data:
            return table_html
        # If there are <th> elements with content, keep the table.
        for header_match in re.finditer(
            r"<th\b[^>]*>(.*?)</th>", table_html, re.IGNORECASE | re.DOTALL
        ):
            header_content = _normalize_cell_text(header_match.group(1))
            if header_content:
                return table_html
        if metrics is not None:
            metrics["tables_dropped"] = metrics.get("tables_dropped", 0) + 1
        return ""

    return TABLE_BLOCK_PATTERN.sub(replace_table, html_text)


def canonicalize_markdown(text: str) -> str:
    text = NBSP_PATTERN.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = DEHYPHEN_PATTERN.sub("", text)
    text = prune_placeholder_cells(text)
    text = drop_empty_tables(text)
    text = CITATION_SUP_PATTERN.sub(lambda m: f"[^{m.group(1)}]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_refdet_regions(text: str) -> List[Tuple[str, List[Tuple[float, float, float, float]]]]:
    regions: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []
    for match in REFDET_EXTRACT_PATTERN.findall(text):
        _, label_text, coords_text = match
        label = label_text.strip().lower()
        if not label:
            continue
        try:
            coords = ast.literal_eval(coords_text)
        except (ValueError, SyntaxError):
            continue
        boxes: List[Tuple[float, float, float, float]] = []
        if isinstance(coords, list):
            for entry in coords:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 4
                    and all(isinstance(v, (int, float)) for v in entry)
                ):
                    boxes.append(tuple(float(v) for v in entry))
        if boxes:
            regions.append((label, boxes))
    return regions


def convert_box_to_pixels(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, int(round(x1 / 999.0 * width))))
    y1 = max(0, min(height, int(round(y1 / 999.0 * height))))
    x2 = max(0, min(width, int(round(x2 / 999.0 * width))))
    y2 = max(0, min(height, int(round(y2 / 999.0 * height))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def collect_roi_jobs(
    job: PageJob,
    regions: List[Tuple[str, List[Tuple[float, float, float, float]]]],
    labels: Sequence[str],
    min_area: int,
    counters: dict[tuple[str, int], int],
) -> List[ROIJob]:
    label_set = {label.lower() for label in labels}
    width, height = job.image.size
    roi_jobs: List[ROIJob] = []
    for label, boxes in regions:
        if label not in label_set:
            continue
        key = (label, job.page_index)
        index_base = counters.get(key, 0)
        added = 0
        for box in boxes:
            x1, y1, x2, y2 = convert_box_to_pixels(box, width, height)
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue
            crop = job.image.crop((x1, y1, x2, y2))
            added += 1
            region_index = index_base + added
            roi_jobs.append(
                ROIJob(
                    pdf_path=job.pdf_path,
                    page_index=job.page_index,
                    region_index=region_index,
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    image=crop,
                )
            )
        counters[key] = index_base + added
    return roi_jobs


@contextlib.contextmanager
def vision_mode_override(base_size: int, image_size: int, crop_mode: bool):
    # DeepSeek OCR processors may not be present in some vLLM releases; fall back to VL2
    try:
        module = importlib.import_module(
            "vllm.transformers_utils.processors.deepseek_ocr"
        )
    except Exception:  # pragma: no cover - capability detection
        module = importlib.import_module(
            "vllm.transformers_utils.processors.deepseek_vl2"
        )
    prev_base = getattr(module, "BASE_SIZE", None)
    prev_image = getattr(module, "IMAGE_SIZE", None)
    prev_crop = getattr(module, "CROP_MODE", None)
    module.BASE_SIZE = base_size
    module.IMAGE_SIZE = image_size
    module.CROP_MODE = crop_mode
    try:
        yield
    finally:
        if prev_base is not None:
            module.BASE_SIZE = prev_base
        if prev_image is not None:
            module.IMAGE_SIZE = prev_image
        if prev_crop is not None:
            module.CROP_MODE = prev_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for combined markdown outputs. Default: <input>/deepseek_vllm_outputs_<mode>.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt for the selected mode.",
    )
    parser.add_argument(
        "--grounded-prompt",
        type=str,
        default=DEFAULT_GROUNDED_PROMPT,
        help="Prompt used in grounded mode (structure + boxes).",
    )
    parser.add_argument(
        "--clean-prompt",
        type=str,
        default=DEFAULT_CLEAN_PROMPT,
        help="Prompt used in clean mode (structure removed).",
    )
    parser.add_argument(
        "--roi-prompt",
        type=str,
        default=DEFAULT_ROI_PROMPT,
        help="Prompt for ROI second-pass crops.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="clean",
        choices=("clean", "grounded"),
        help="Select output profile. Run twice (grounded & clean) to get both artifacts.",
    )
    parser.add_argument(
        "--batch-pages",
        type=int,
        default=12,
        help="Number of pages to batch per vLLM generate() call.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to decode per page.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Rendering DPI for PDF pages.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=8,
        help="Thread pool size for PDF rendering.",
    )
    # Default model: prefer local checkout if present; otherwise pick a remote id.
    default_model = (
        str(CHECKPOINT_DIR)
        if CHECKPOINT_DIR.exists()
        else ("deepseek-ai/DeepSeek-OCR" if _DEEPSEEK_OCR_AVAILABLE else "deepseek-ai/DeepSeek-VL2")
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Model identifier or local path.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Precision for model weights/activations.",
    )
    parser.add_argument(
        "--no-fp8-kv",
        action="store_true",
        help="Disable FP8 KV cache.",
    )
    parser.add_argument(
        "--enable-fp8-weights",
        action="store_true",
        help="Enable FP8 weight quantization (verify quality).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page cap per PDF.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Target fraction of GPU memory for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size to pass to vLLM when spanning multiple GPUs.",
    )
    parser.add_argument(
        "--mm-encoder-tp-mode",
        type=str,
        choices=("auto", "data", "sequence"),
        default="auto",
        help="Tensor-parallel strategy for the multimodal encoder (use 'data' for data-parallel fan-out).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pages whose outputs already exist.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the PDF list into this many shards for multi-GPU runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based index of the shard to process (requires --num-shards > 1).",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Persist rendered page PNGs alongside markdown outputs.",
    )
    parser.add_argument(
        "--roi-second-pass",
        action="store_true",
        help="After grounded run, crop specified regions and re-run clean inference per crop.",
    )
    parser.add_argument(
        "--roi-label",
        action="append",
        default=None,
        help="Region label to include in ROI second pass (repeatable). Defaults to table/title/paragraph/figure.",
    )
    parser.add_argument(
        "--roi-min-area",
        type=int,
        default=2048,
        help="Minimum pixel area for ROI crops in second pass.",
    )
    parser.add_argument(
        "--retry-large",
        action="store_true",
        help="Re-run pages with missing ref/det matches using Large (1280) vision mode.",
    )
    parser.add_argument(
        "--retry-label",
        action="append",
        default=None,
        help="Region label that must appear; otherwise trigger Large retry (default: table).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--content-debug",
        action="store_true",
        help="Include page separators (---pages---) and truncation markers.",
    )
    return parser.parse_args()


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.mode == "grounded":
        return args.grounded_prompt
    return args.clean_prompt


def process_batch(
    llm: LLM,
    batch: Sequence[PageJob],
    params: SamplingParams,
    prompt: str,
    vision_override: Optional[Tuple[int, int, bool]] = None,
) -> List[Tuple[PageJob, str, int, bool]]:
    requests = [
        {"prompt": prompt, "multi_modal_data": {"image": job.image}} for job in batch
    ]
    manager = (
        vision_mode_override(*vision_override)
        if vision_override is not None
        else contextlib.nullcontext()
    )
    with manager:
        outputs = llm.generate(requests, params)
    result_tuples: List[Tuple[PageJob, str, int, bool]] = []
    for job, output in zip(batch, outputs):
        generated = output.outputs[0]
        text = generated.text
        token_ids = getattr(generated, "token_ids", ())
        token_count = len(token_ids)
        finish_reason = getattr(generated, "finish_reason", None)
        token_limit_hit = False
        if finish_reason is not None:
            reason_text = str(finish_reason).lower()
            token_limit_hit = "length" in reason_text or "max_token" in reason_text
        if not token_limit_hit:
            max_tokens = getattr(params, "max_tokens", None)
            if max_tokens is not None and token_count >= max_tokens:
                token_limit_hit = True
        logging.info(
            "Decoded %s#%04d: %d tokens, %d chars (preview=%r)",
            job.pdf_path.name,
            job.page_index + 1,
            token_count,
            len(text),
            text[:80],
        )
        result_tuples.append((job, text, token_count, token_limit_hit))
    return result_tuples


def prepare_page_text(
    job: PageJob,
    text: str,
    *,
    keep_refdet: bool,
    prompt: str,
    token_limit_hit: bool = False,
    token_limit: Optional[int] = None,
    metrics: Optional[dict[str, int]] = None,
    content_debug: bool = False,
) -> str:
    text = strip_prompt_echo(text, prompt)
    cleaned = clean_output(text, keep_refdet=keep_refdet, metrics=metrics)
    logging.debug(
        "Prepared page %s#%04d (%d chars)",
        job.pdf_path.name,
        job.page_index + 1,
        len(cleaned),
    )
    if LEFTOVER_META_PATTERN.search(cleaned):
        if metrics is not None:
            metrics["residual_meta_pages"] = (
                metrics.get("residual_meta_pages", 0) + 1
            )
        logging.warning(
            "[%s] residual meta token markers detected on page %d",
            job.pdf_path.name,
            job.page_index + 1,
        )
    try:
        with open("/tmp/debug.txt", "a", encoding="utf-8") as dbg:
            dbg.write(f"{job.pdf_path.name}:{job.page_index+1} len={len(cleaned)} raw_len={len(text)}\n")
    except OSError:
        pass
    if token_limit_hit and content_debug:
        warning = (
            f"[[Token limit reached at {token_limit} tokens; page may be truncated]]"
            if token_limit
            else "[[Token limit reached; page may be truncated]]"
        )
        cleaned = f"{cleaned.rstrip()}\n\n{warning}" if cleaned else warning
        if metrics is not None:
            metrics["token_limit_hits"] = metrics.get("token_limit_hits", 0) + 1
    return cleaned


def get_blank_page_text(placeholder: str = "[[Blank page]]") -> str:
    return placeholder.strip()


def stash_page_image(job: PageJob, assets_root: Optional[Path]) -> None:
    if assets_root is None:
        return
    page_dir = assets_root / f"page_{job.page_index+1:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    image_path = page_dir / "page.png"
    if not image_path.exists():
        job.image.save(image_path, format="PNG", optimize=True)


def run_roi_second_pass(
    llm: LLM,
    roi_jobs: List[ROIJob],
    params: SamplingParams,
    prompt: str,
    assets_root: Optional[Path],
    metrics: Optional[dict[str, int]] = None,
) -> Tuple[int, int]:
    if not roi_jobs or assets_root is None:
        return 0, 0
    total_tokens = 0
    for batch in batched(roi_jobs, max(1, min(len(roi_jobs), 8))):
        requests = [
            {"prompt": prompt, "multi_modal_data": {"image": roi.image}}
            for roi in batch
        ]
        outputs = llm.generate(requests, params)
        for roi, output in zip(batch, outputs):
            text = output.outputs[0].text
            total_tokens += len(output.outputs[0].token_ids)
            cleaned = clean_output(text, keep_refdet=False, metrics=metrics)
            page_dir = assets_root / f"page_{roi.page_index+1:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            roi_path = page_dir / f"roi_{roi.label}_{roi.region_index:02d}.md"
            header = f"<!-- bbox=({roi.bbox[0]},{roi.bbox[1]},{roi.bbox[2]},{roi.bbox[3]}) -->\n"
            roi_path.write_text(header + cleaned, encoding="utf-8")
    return len(roi_jobs), total_tokens


def write_combined_markdown(
    combined_path: Path, aggregated_pages: dict[int, str], *, content_debug: bool = False
) -> None:
    if not aggregated_pages:
        return

    sorted_pages = sorted(aggregated_pages)
    sections: List[str] = []
    for offset, page_index in enumerate(sorted_pages):
        page_text = aggregated_pages[page_index].strip()
        if not page_text:
            continue
        if content_debug and offset > 0:
            sections.append("---pages---")
        sections.append(page_text)

    body = "\n\n".join(sections).strip()
    if body:
        body = canonicalize_markdown(body)
    combined = f"{body}\n" if body else ""
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(combined, encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    ensure_cuda_visible()

    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be >= 1.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard < num_shards.")

    prompt = resolve_prompt(args)
    roi_labels = tuple(
        label.lower()
        for label in (
            args.roi_label
            if args.roi_label is not None
            else ("table", "title", "paragraph", "figure", "equation", "list")
        )
    )
    retry_labels = tuple(
        label.lower()
        for label in (args.retry_label if args.retry_label else DEFAULT_RETRY_LABELS)
    )

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    default_dir = (
        "deepseek_vllm_outputs_grounded"
        if args.mode == "grounded"
        else "deepseek_vllm_outputs_clean"
    )
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / default_dir
    )
    output_root.mkdir(parents=True, exist_ok=True)

    all_pdf_files = sorted(f for f in input_dir.glob("*.pdf") if f.is_file())
    if not all_pdf_files:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")
    if args.num_shards > 1:
        pdf_files = all_pdf_files[args.shard_index :: args.num_shards]
        logging.info(
            "Shard %d/%d assigned %d of %d PDF(s).",
            args.shard_index,
            args.num_shards,
            len(pdf_files),
            len(all_pdf_files),
        )
        if not pdf_files:
            logging.warning(
                "Shard %d/%d has no PDFs to process; exiting.",
                args.shard_index,
                args.num_shards,
            )
            return
    else:
        pdf_files = all_pdf_files

    whitelist_ids = compute_whitelist_token_ids()
    llm = None
    try:
        llm = build_llm(args)
        sampling_params = build_sampling_params(args, args.mode, whitelist_ids)
    clean_params = (
        build_sampling_params(args, "clean", whitelist_ids)
        if args.roi_second_pass or args.mode == "clean"
        else None
    )

    total_pages = 0
    total_tokens = 0
    roi_total_tokens = 0
    run_start = time.perf_counter()
    metrics: dict[str, int] = {}

        executor = ThreadPoolExecutor(max_workers=args.cpu_workers)
        try:
            for pdf_path in pdf_files:
            combined_name = f"{pdf_path.stem}.md"
            combined_path = output_root / combined_name
            if args.skip_existing and combined_path.exists():
                logging.info(
                    "[%s] combined output already exists; skipping.",
                    pdf_path.name,
                )
                continue

            page_jobs = render_pdf(pdf_path, args.dpi, executor, args.max_pages)
            if not page_jobs:
                continue

            assets_root: Optional[Path] = None
            if args.save_images or args.roi_second_pass:
                assets_root = output_root / f"{pdf_path.stem}_assets"
                assets_root.mkdir(parents=True, exist_ok=True)

            aggregated_pages: dict[int, str] = {}
            pdf_start = time.perf_counter()
            pages_written = 0
            roi_jobs: List[ROIJob] = []
            roi_counters: dict[tuple[str, int], int] = {}
            pages_to_retry: List[PageJob] = []

            page_token_limit = getattr(sampling_params, "max_tokens", None)
            for batch in batched(page_jobs, args.batch_pages):
                working_batch = []
                for job in batch:
                    if job.is_blank:
                        blank_text = get_blank_page_text()
                        aggregated_pages[job.page_index] = (
                            blank_text if args.content_debug else ""
                        )
                        pages_written += 1
                        if args.save_images:
                            stash_page_image(job, assets_root)
                        logging.debug(
                            "[%s] detected blank page %d; skipping inference",
                            pdf_path.name,
                            job.page_index + 1,
                        )
                        continue
                    working_batch.append(job)
                if not working_batch:
                    continue
                results = process_batch(
                    llm, working_batch, sampling_params, prompt, BASE_VISION_CONFIG
                )
                for job, raw_text, token_count, token_limit_hit in results:
                    keep_refdet = args.mode == "grounded"
                    cleaned = prepare_page_text(
                        job,
                        raw_text,
                        keep_refdet=keep_refdet,
                        prompt=prompt,
                        token_limit_hit=token_limit_hit,
                        token_limit=page_token_limit,
                        metrics=metrics,
                        content_debug=bool(args.content_debug),
                    )
                    aggregated_pages[job.page_index] = cleaned
                    if args.save_images:
                        stash_page_image(job, assets_root)

                    total_tokens += token_count
                    pages_written += 1

                    if keep_refdet:
                        regions = extract_refdet_regions(raw_text)
                        label_counts = {label: len(boxes) for label, boxes in regions}
                        if args.retry_large and any(
                            label_counts.get(label, 0) == 0 for label in retry_labels
                        ):
                            pages_to_retry.append(job)
                        if args.roi_second_pass:
                            roi_jobs.extend(
                                collect_roi_jobs(
                                    job,
                                    regions,
                                    roi_labels,
                                    args.roi_min_area,
                                    roi_counters,
                                )
                            )

            if args.retry_large and pages_to_retry:
                logging.info(
                    "[%s] retrying %d page(s) in Large vision mode",
                    pdf_path.name,
                    len(pages_to_retry),
                )
                if args.roi_second_pass:
                    retry_indices = {job.page_index for job in pages_to_retry}
                    roi_jobs = [
                        roi for roi in roi_jobs if roi.page_index not in retry_indices
                    ]
                    for key in list(roi_counters.keys()):
                        if key[1] in retry_indices:
                            roi_counters.pop(key, None)
                for batch in batched(pages_to_retry, args.batch_pages):
                    results_large = process_batch(
                        llm, batch, sampling_params, prompt, LARGE_VISION_CONFIG
                    )
                    for job, raw_text, token_count, token_limit_hit in results_large:
                        keep_refdet = args.mode == "grounded"
                        cleaned = prepare_page_text(
                            job,
                            raw_text,
                            keep_refdet=keep_refdet,
                            prompt=prompt,
                            token_limit_hit=token_limit_hit,
                            token_limit=page_token_limit,
                            metrics=metrics,
                            content_debug=bool(args.content_debug),
                        )
                        aggregated_pages[job.page_index] = cleaned
                        total_tokens += token_count
                        if args.save_images:
                            stash_page_image(job, assets_root)
                        if keep_refdet and args.roi_second_pass:
                            regions = extract_refdet_regions(raw_text)
                            roi_jobs.extend(
                                collect_roi_jobs(
                                    job,
                                    regions,
                                    roi_labels,
                                    args.roi_min_area,
                                    roi_counters,
                                )
                            )

            if args.roi_second_pass:
                if args.mode != "grounded":
                    logging.warning(
                        "[%s] ROI second pass requires grounded mode; skipping.",
                        pdf_path.name,
                    )
                elif not roi_jobs:
                    logging.info("[%s] No ROI targets discovered.", pdf_path.name)
                elif clean_params is None:
                    logging.warning(
                        "[%s] Clean sampling params unavailable; skipping ROI.",
                        pdf_path.name,
                    )
                else:
                    roi_count, roi_tokens = run_roi_second_pass(
                        llm,
                        roi_jobs,
                        clean_params,
                        args.roi_prompt,
                        assets_root,
                        metrics,
                    )
                    roi_total_tokens += roi_tokens
                    logging.info(
                        "[%s] ROI second pass completed for %d region(s).",
                        pdf_path.name,
                        roi_count,
                    )

            write_combined_markdown(
                combined_path, aggregated_pages, content_debug=bool(args.content_debug)
            )

            total_pages += pages_written
            pdf_elapsed = time.perf_counter() - pdf_start
            if pages_written:
                logging.info(
                    "[%s] processed %d page(s) in %.1fs (%.2f pp/s)",
                    pdf_path.name,
                    pages_written,
                    pdf_elapsed,
                    pages_written / max(pdf_elapsed, 1e-6),
                )

        finally:
            executor.shutdown(wait=True)

        total_elapsed = time.perf_counter() - run_start
        pages_per_sec = total_pages / max(total_elapsed, 1e-6)
        tokens_per_sec = total_tokens / max(total_elapsed, 1e-6)
        logging.info(
            "Completed %d page(s) in %.1fs (%.2f pages/s, %d tokens/s, %d ROI tokens)",
            total_pages,
            total_elapsed,
            pages_per_sec,
            int(tokens_per_sec),
            roi_total_tokens,
        )
        metric_parts: List[str] = []
        if metrics:
            metric_parts = [f"{key}={value}" for key, value in metrics.items() if value]
        if metric_parts:
            logging.info("Sanity metrics: %s", ", ".join(sorted(metric_parts)))
    finally:
        # Encourage vLLM to shut down GPU/engine threads so process exits cleanly
        with contextlib.suppress(Exception):
            engine = getattr(llm, "llm_engine", None) if llm is not None else None
            if engine is not None:
                shutdown = getattr(engine, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        # Best-effort cleanup
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc as _gc

        del llm
        _gc.collect()


if __name__ == "__main__":
    main()
