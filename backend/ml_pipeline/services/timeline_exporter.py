"""Helpers for exporting cleaned timeline text for review and auditing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

from django.conf import settings


def sanitize_handle(handle: str) -> str:
    """Convert an X handle into a safe filename stem."""
    clean = (handle or '').strip().lstrip('@')
    clean = re.sub(r'[^A-Za-z0-9_.-]+', '_', clean)
    return clean or 'unknown_handle'


def export_cleaned_posts_to_txt(handle: str, cleaned_posts: Iterable[str]) -> Path:
    """
    Export cleaned post text to a handle-named .txt file.

    The file is overwritten on each run so the latest timeline is always visible.
    """
    export_dir = Path(getattr(settings, 'TIMELINE_EXPORT_DIR', Path(settings.BASE_DIR) / 'exports' / 'timelines'))
    export_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{sanitize_handle(handle)}.txt"
    export_path = export_dir / filename

    lines: list[str] = []
    for index, post in enumerate(cleaned_posts, start=1):
        text = (post or '').strip()
        if not text:
            continue
        lines.append(f"{index}. {text}")

    export_path.write_text("\n\n".join(lines) + ("\n" if lines else ""), encoding='utf-8')
    return export_path

