"""Probe the TAAC2026 demo dataset and print its structure.

Usage:
    python scripts/probe_demo_dataset.py

Notes:
- Reads directly from HuggingFace via pandas (no local cache needed for the parquet file).
- Requires: pandas, pyarrow, fsspec, huggingface_hub
"""
import re

import numpy as np
import pandas as pd


URL = "hf://datasets/TAAC2026/data_sample_1000/demo_1000.parquet"


def main() -> None:
    df = pd.read_parquet(URL)
    print(f"shape = {df.shape}")

    groups: dict[str, list[str]] = {
        "meta": ["user_id", "item_id", "label_type", "label_time", "timestamp"],
        "user_int_feats": [c for c in df.columns if c.startswith("user_int_feats_")],
        "user_dense_feats": [c for c in df.columns if c.startswith("user_dense_feats_")],
        "item_int_feats": [c for c in df.columns if c.startswith("item_int_feats_")],
    }
    for dom in ("a", "b", "c", "d"):
        groups[f"domain_{dom}_seq"] = [c for c in df.columns if c.startswith(f"domain_{dom}_seq_")]

    for gname, cols in groups.items():
        print(f"\n--- {gname} ({len(cols)} cols) ---")
        for c in cols:
            s = df[c]
            nn = s.dropna()
            if len(nn) == 0:
                print(f"  {c}: ALL NULL")
                continue
            v = nn.iloc[0]
            if isinstance(v, np.ndarray):
                info = f"ndarray(shape={v.shape}, dtype={v.dtype})"
            else:
                info = f"{type(v).__name__}"
            print(f"  {c}: null={s.isna().mean():.1%} dtype={s.dtype} sample={info}")

    # Sequence length stats per domain (lengths are consistent across all columns within a domain on the same row).
    print("\n=== per-domain sequence length distribution ===")
    for dom in ("a", "b", "c", "d"):
        col = f"domain_{dom}_seq_{{'a':38,'b':67,'c':27,'d':17}[dom]}"
        # use eval-free mapping
        col = {"a": "domain_a_seq_38", "b": "domain_b_seq_67",
               "c": "domain_c_seq_27", "d": "domain_d_seq_17"}[dom]
        lens = df[col].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
        print(f"  domain_{dom}: min={lens.min()} p50={int(lens.median())} "
              f"p95={int(lens.quantile(0.95))} max={lens.max()} mean={lens.mean():.1f}")

    print("\n=== label_type distribution ===")
    print(df["label_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
