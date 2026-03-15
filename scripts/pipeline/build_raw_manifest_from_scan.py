#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def parse_scan_spec(spec: str) -> Dict[str, str]:
    item = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid token in --scan: {token}. Expected key=value")
        k, v = token.split("=", 1)
        item[k.strip()] = v.strip()

    required = ["source", "split", "format", "path"]
    for k in required:
        if k not in item:
            raise ValueError(f"Missing {k} in --scan spec: {spec}")

    fmt = item["format"]
    if fmt not in {"npy", "smplx_pkl_dir"}:
        raise ValueError(f"Unsupported format={fmt}. Use npy or smplx_pkl_dir")

    if "glob" not in item:
        item["glob"] = "**/*.npy" if fmt == "npy" else "*"

    if "layout" not in item:
        # npy can be soke133 or smplx179. pkl-dir implies smplx179.
        item["layout"] = "soke133" if fmt == "npy" else "smplx179"

    return item


def expand_scan(spec: Dict[str, str]) -> List[Dict]:
    root = Path(spec["path"]).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Scan root does not exist: {root}")

    pattern = spec["glob"]
    matched = list(root.glob(pattern))

    out = []
    for p in matched:
        if spec["format"] == "npy":
            if not p.is_file() or p.suffix != ".npy":
                continue
        else:
            if not p.is_dir():
                continue

        name = p.stem if p.is_file() else p.name
        out.append(
            {
                "path": str(p),
                "source": spec["source"],
                "split": spec["split"],
                "format": spec["format"],
                "layout": spec.get("layout", "soke133"),
                "name": name,
            }
        )

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build raw-manifest JSONL by scanning local folders for npy files or SMPL-X pkl sequence directories."
    )
    parser.add_argument(
        "--scan",
        action="append",
        required=True,
        help=(
            "Repeatable scan spec. Example: "
            "source=motionx,split=train,format=npy,path=/data/motionx,glob=**/*.npy,layout=smplx179"
        ),
    )
    parser.add_argument("--output", required=True, help="Output raw manifest jsonl path")
    parser.add_argument("--sort", action="store_true", help="Sort output by path")
    args = parser.parse_args()

    items: List[Dict] = []
    for spec_str in args.scan:
        spec = parse_scan_spec(spec_str)
        expanded = expand_scan(spec)
        items.extend(expanded)
        print(f"[scan] source={spec['source']} split={spec['split']} matched={len(expanded)}")

    if args.sort:
        items.sort(key=lambda x: x["path"])

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} entries to {output}")


if __name__ == "__main__":
    main()
