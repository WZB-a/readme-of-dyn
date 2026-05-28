from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.data import build_pair_index
from libero_pi_dyn.data import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = ATGConfig.from_file(args.config)
    root = Path(config.data.cache_root)
    for split in (config.data.train_split, config.data.val_split):
        rows = build_pair_index(config, split)
        out = root / "pair_index" / f"{split}.jsonl"
        write_jsonl(out, rows)
        print(f"{split}: wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
