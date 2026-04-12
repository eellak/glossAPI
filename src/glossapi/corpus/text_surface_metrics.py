from __future__ import annotations

import re

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
INLINE_DISPLAY_MATH_RE = re.compile(r"(\\\[.*?\\\])|(\\\(.*?\\\))|(\$\$.*?\$\$)", re.S)
CHAR_COUNT_LATEX_ENV_NAMES = (
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "comment",
)


def _strip_latex_envs_for_char_count(text: str) -> str:
    cleaned = text
    for env in CHAR_COUNT_LATEX_ENV_NAMES:
        escaped = re.escape(env)
        cleaned = re.sub(
            rf"\\begin\{{{escaped}\}}.*?\\end\{{{escaped}\}}",
            "",
            cleaned,
            flags=re.S,
        )
    return cleaned


def sanitized_char_count(content: str) -> tuple[int, bool]:
    """Return export-facing non-whitespace char count and emptiness for text.

    The cleaner, export-facing metadata refresh, and OpenArchives patching must
    all agree on this contract so they describe the exact published text
    surface.
    """

    sanitized = HTML_COMMENT_RE.sub("", content)
    sanitized = _strip_latex_envs_for_char_count(sanitized)
    sanitized = INLINE_DISPLAY_MATH_RE.sub("", sanitized)
    count = sum(1 for ch in sanitized if not ch.isspace())
    return count, count == 0

