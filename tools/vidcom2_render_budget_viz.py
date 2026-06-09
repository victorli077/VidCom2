#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Set

from token_compressor.vidcom2.visualization import render_budget_from_metadata


def parse_case_selector(raw: Optional[str]) -> Optional[Set[int]]:
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in {"", "all", "*"}:
        return None
    out: Set[int] = set()
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(p))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render VidCom2 budget comparison figures from saved metadata json files."
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="logs/vidcom2_budget_viz",
        help="Directory containing *_budget_compare.json artifacts.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help='Case filter, e.g. "all", "0,3,5-8".',
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="Maximum number of cases to render (0 means no limit).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="WorldSense Budget: Visual-only vs Audio-guided",
        help="Figure title.",
    )
    args = parser.parse_args()

    viz_dir = Path(args.viz_dir)
    if not viz_dir.exists():
        print(f"[vidcom2_render_budget_viz] directory not found: {viz_dir}")
        return 1

    selected_cases = parse_case_selector(args.cases)
    metas = sorted(viz_dir.glob("*_budget_compare.json"))
    rendered = 0
    for meta_path in metas:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            case_idx = int(meta.get("case_index", -1))
            if selected_cases is not None and case_idx not in selected_cases:
                continue
            out = render_budget_from_metadata(
                meta_path=str(meta_path),
                out_path=None,
                title=args.title,
            )
            if out is not None:
                rendered += 1
                print(f"[vidcom2_render_budget_viz] rendered: {out}")
            if args.max_cases > 0 and rendered >= args.max_cases:
                break
        except Exception as e:
            print(f"[vidcom2_render_budget_viz] failed on {meta_path}: {e}")

    print(f"[vidcom2_render_budget_viz] done, rendered {rendered} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

