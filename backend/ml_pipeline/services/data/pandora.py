"""PANDORA data ingestion and conversion helpers.

The cloned PANDORA repository stores parquet shards under:

    PANDORA/pandora-big5/data/

The converted files used in this project expose rows with:

    O, C, E, A, N, ptype, text, __index_level_0__

This module is the single source of truth for locating those parquet shards,
loading them, cleaning them with the shared text cleaner, and converting them
into row-level training records.

Important:
    - This snapshot behaves like comment-level data.
    - Do not group by trait tuple and treat that as identity for training.
    - Keep BFI/OCEAN scores as labels, not as a user key.
    - If an upstream PANDORA release contains true author ids, this module
      can be extended to support author-level grouping without changing the
      downstream training interface.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.ml_pipeline.cleaning.cleaner import CleanedContent, DataCleaner, RawXData

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("backend/ml_pipeline/data/prepared/pandora_prepared.json")
DEFAULT_PANDORA_DATA_DIR = Path("PANDORA/pandora-big5/data")

_TRAIT_COLS = ("O", "C", "E", "A", "N")


@dataclass
class PandoraTraits:
    """Big Five percentile scores plus PANDORA class label."""

    O: float  # noqa: E741 - dataset column name
    C: float
    E: float
    A: float
    N: float
    ptype: int


@dataclass
class PandoraRecord:
    """Single cleaned PANDORA row in the project's internal training shape."""

    sample_id: str
    text: str
    traits: PandoraTraits
    source_file: str
    source_row: int
    split_hint: str = "train"
    source_path: str | None = None

    def to_serializable(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "traits": asdict(self.traits),
            "source_file": self.source_file,
            "source_row": self.source_row,
            "split_hint": self.split_hint,
            "source_path": self.source_path,
        }


def discover_pandora_parquet_files(
    base_dir: str | Path = DEFAULT_PANDORA_DATA_DIR,
) -> list[Path]:
    """Return all parquet shards found in the cloned PANDORA data folder."""
    root = Path(base_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.parquet") if path.is_file())


def resolve_default_pandora_source(
    explicit_path: str | Path | None = None,
) -> Path:
    """Resolve the preferred PANDORA source path."""
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"PANDORA source not found: {path}")
        return path

    discovered = discover_pandora_parquet_files()
    if not discovered:
        raise FileNotFoundError(
            f"No parquet files were found in {DEFAULT_PANDORA_DATA_DIR}."
        )

    if len(discovered) == 1:
        return discovered[0]

    return DEFAULT_PANDORA_DATA_DIR


def _iter_pandora_files(pandora_source: str | Path) -> list[Path]:
    path = Path(pandora_source)
    if path.is_dir():
        files = discover_pandora_parquet_files(path)
        if not files:
            raise FileNotFoundError(f"No parquet shards found in {path}")
        return files
    if path.suffix.lower() == ".parquet":
        if not path.exists():
            raise FileNotFoundError(f"PANDORA file not found: {path}")
        return [path]
    raise ValueError(f"Unsupported PANDORA source: {path}")


def _load_raw_rows(
    pandora_source: str | Path,
) -> Iterable[tuple[str, int, dict[str, Any]]]:
    """
    Stream rows out of the PANDORA parquet export.

    Requires pandas + pyarrow.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading the PANDORA parquet export requires pandas and pyarrow: "
            "pip install pandas pyarrow"
        ) from exc

    for file_path in _iter_pandora_files(pandora_source):
        df = pd.read_parquet(file_path, columns=[*_TRAIT_COLS, "ptype", "text"])
        for row_idx, row in enumerate(df.to_dict(orient="records")):
            yield file_path.name, row_idx, row


def _build_raw_payload(
    rows: Iterable[tuple[str, int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tweets: list[dict[str, Any]] = []
    metadata_index: list[dict[str, Any]] = []

    for source_file, source_row, row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        tweets.append(
            {
                "text": text,
                "id_str": f"{source_file}:{source_row}",
                "source_file": source_file,
                "source_row": source_row,
                "source_path": str(Path(source_file)),
                "O": row.get("O"),
                "C": row.get("C"),
                "E": row.get("E"),
                "A": row.get("A"),
                "N": row.get("N"),
                "ptype": row.get("ptype"),
            }
        )
        metadata_index.append(
            {
                "source_file": source_file,
                "source_row": source_row,
                "source_path": str(Path(source_file)),
                "traits": row,
            }
        )

    return tweets, metadata_index


def _row_to_traits(row: dict[str, Any]) -> PandoraTraits | None:
    try:
        return PandoraTraits(
            O=float(row["O"]),
            C=float(row["C"]),
            E=float(row["E"]),
            A=float(row["A"]),
            N=float(row["N"]),
            ptype=int(row["ptype"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _clean_records(
    pandora_source: str | Path,
    min_text_length: int = 3,
) -> list[PandoraRecord]:
    rows = list(_load_raw_rows(pandora_source))
    tweets, _ = _build_raw_payload(rows)

    raw = RawXData(profile={}, tweets=tweets, replies=[])
    cleaner = DataCleaner(min_text_length=min_text_length)
    cleaned_items = cleaner.clean(raw)

    by_id: dict[str, dict[str, Any]] = {}
    for source_file, source_row, row in rows:
        by_id[f"{source_file}:{source_row}"] = {
            "source_file": source_file,
            "source_row": source_row,
            "row": row,
        }

    records: list[PandoraRecord] = []
    for item in cleaned_items:
        if item.content_type != "tweet":
            continue
        meta = by_id.get(item.content_id.replace("tweet_", "", 1))
        if meta is None:
            source_file = item.metadata.get("source_file") if item.metadata else "unknown"
            source_row = int(item.metadata.get("source_row") or -1) if item.metadata else -1
            row = {}
        else:
            source_file = meta["source_file"]
            source_row = int(meta["source_row"])
            row = meta["row"]

        traits = _row_to_traits(row)
        if traits is None:
            continue

        split_hint = "train"
        if "validation" in source_file.lower():
            split_hint = "validation"
        elif "test" in source_file.lower():
            split_hint = "test"

        records.append(
            PandoraRecord(
                sample_id=f"{source_file}:{source_row}",
                text=item.cleaned_text,
                traits=traits,
                source_file=source_file,
                source_row=source_row,
                split_hint=split_hint,
                source_path=str(Path(source_file)),
            )
        )

    return records


def save_flat_pandora_records(
    records: list[PandoraRecord],
    output_path: str | Path,
) -> None:
    """Persist the flattened PANDORA dataset as JSON for downstream training."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with path.open("w", encoding="utf-8") as fh:
        json.dump([r.to_serializable() for r in records], fh, ensure_ascii=False, indent=2)
    logger.info("Saved flattened PANDORA records to %s", path)


def load_pandora_records(
    pandora_source: str | Path | None = None,
    *,
    min_text_length: int = 3,
    output_path: str | Path | None = None,
) -> list[PandoraRecord]:
    """
    Load the PANDORA parquet shards and return flattened training records.

    This is the preferred entry point for the new training pipeline.
    It keeps the dataset in sample-level form, which is the correct match
    for the converted parquet snapshot found in the cloned repository.
    """
    source = resolve_default_pandora_source(pandora_source)
    records = _clean_records(source, min_text_length=min_text_length)

    if output_path is not None:
        save_flat_pandora_records(records, output_path)

    logger.info("Loaded %d flattened PANDORA records from %s", len(records), source)
    return records


def load_pandora_comments(
    pandora_source: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_text_length: int = 3,
) -> list[PandoraRecord]:
    """
    Backward-compatible wrapper for older code paths.

    The new project architecture uses row-level records, so this now
    returns the same flattened records as load_pandora_records().
    """
    return load_pandora_records(
        pandora_source=pandora_source,
        min_text_length=min_text_length,
        output_path=output_path,
    )
  
