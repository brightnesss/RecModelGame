"""Print one sample row from the TAAC2026 demo dataset in a human-readable layout.

Usage:
    python scripts/show_one_sample.py [ROW_INDEX]

Default ROW_INDEX = 0. Requires pandas, pyarrow, fsspec, huggingface_hub.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd


URL = "hf://datasets/TAAC2026/data_sample_1000/demo_1000.parquet"


def fmt_value(v) -> str:
    """Compactly format a cell value for display."""
    if v is None:
        return "None"
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "f":
            preview = np.array2string(v[:6], precision=4, suppress_small=True)
            return f"ndarray[float{v.dtype.itemsize*8}] shape={v.shape} first6={preview}"
        preview = v[:10].tolist()
        return f"ndarray[{v.dtype}] len={len(v)} first10={preview}"
    return f"{type(v).__name__}={v}"


def group_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def show_flat(title: str, row: pd.Series, cols: list[str], limit: int | None = None) -> None:
    print(f"\n[{title}]  ({len(cols)} 列)")
    shown = cols if limit is None else cols[:limit]
    for c in shown:
        print(f"  {c:28s} : {fmt_value(row[c])}")
    if limit is not None and len(cols) > limit:
        print(f"  ... (+{len(cols) - limit} 列省略)")


def show_sequence_domain(domain: str, row: pd.Series, cols: list[str]) -> None:
    print(f"\n[domain_{domain}_seq]  ({len(cols)} 列并列描述同一条序列)")
    length = None
    for c in cols:
        v = row[c]
        if isinstance(v, np.ndarray):
            length = len(v)
            break
    print(f"  本行该域序列长度 = {length}")
    for c in cols:
        v = row[c]
        if not isinstance(v, np.ndarray):
            print(f"  {c:22s} : None")
            continue
        head = v[:5].tolist()
        tail = v[-3:].tolist() if len(v) > 5 else []
        print(f"  {c:22s} : first5={head}  last3={tail}")


def main() -> None:
    row_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    df = pd.read_parquet(URL)
    if not (0 <= row_index < len(df)):
        raise SystemExit(f"row_index {row_index} out of range [0, {len(df)})")
    row = df.iloc[row_index]

    print("=" * 70)
    print(f"Sample row #{row_index}  (共 {df.shape[0]} 行, {df.shape[1]} 列)")
    print("=" * 70)

    print("\n[Meta / Label]")
    for c in ("user_id", "item_id", "label_type", "label_time", "timestamp"):
        extra = ""
        if c in ("label_time", "timestamp"):
            extra = f"  ({dt.datetime.utcfromtimestamp(int(row[c])).strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        print(f"  {c:15s} = {row[c]}{extra}")
    print(f"  label_time - timestamp = {int(row['label_time']) - int(row['timestamp'])} s")

    show_flat("User int feats (单值+多值稀疏)", row, group_cols(df, "user_int_feats_"), limit=10)
    show_flat("User dense feats (稠密 embedding)", row, group_cols(df, "user_dense_feats_"))
    show_flat("Item int feats (广告/物品侧)", row, group_cols(df, "item_int_feats_"))

    for domain in ("a", "b", "c", "d"):
        show_sequence_domain(domain, row, group_cols(df, f"domain_{domain}_seq_"))


if __name__ == "__main__":
    main()
