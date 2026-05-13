from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("/data6/wenchangxi/community_note/analysis/llm_16agent_rawrating_2000_20260512_officialschema"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data6/wenchangxi/community_note/analysis/llm_16agent_rawrating_balanced_1to1to1_20260513"),
    )
    parser.add_argument("--per-class", type=int, default=0, help="0 means use the smallest available class.")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--run-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = args.source_run_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(src / "note_vote_summary.csv", low_memory=False)
    summary["noteId"] = summary["noteId"].astype(str)
    counts = summary["true_label_text"].value_counts().to_dict()
    labels = ["NOT_HELPFUL", "NEEDS_MORE_RATINGS", "HELPFUL"]
    missing = [label for label in labels if label not in counts]
    if missing:
        raise RuntimeError(f"Missing labels in source summary: {missing}; counts={counts}")
    per_class = args.per_class or min(int(counts[label]) for label in labels)

    selected_parts = []
    for label in labels:
        group = summary[summary["true_label_text"] == label]
        selected_parts.append(group.sample(n=per_class, random_state=args.seed))
    selected_summary = (
        pd.concat(selected_parts, ignore_index=True)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )
    selected_ids = set(selected_summary["noteId"].astype(str))

    votes = pd.read_csv(src / "agent_votes.csv", low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    selected_votes = votes[votes["noteId"].isin(selected_ids)].copy()

    pilot = pd.read_csv(src / "pilot_notes.csv", low_memory=False)
    pilot["noteId"] = pilot["noteId"].astype(str)
    selected_pilot = pilot[pilot["noteId"].isin(selected_ids)].copy()

    selected_summary.to_csv(out / "note_vote_summary.csv", index=False, encoding="utf-8-sig")
    selected_votes.to_csv(out / "agent_votes.csv", index=False, encoding="utf-8-sig")
    selected_pilot.to_csv(out / "pilot_notes.csv", index=False, encoding="utf-8-sig")
    for name in ["persona_prompts.csv", "run_metadata.json"]:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    metadata = {
        "source_run_dir": str(src),
        "out_dir": str(out),
        "seed": args.seed,
        "source_true_distribution": counts,
        "per_class": per_class,
        "n_notes": int(len(selected_summary)),
        "n_agent_votes": int(len(selected_votes)),
        "balanced_true_distribution": selected_summary["true_label_text"].value_counts().to_dict(),
    }
    (out / "balanced_sample_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    if args.run_eval:
        script = Path("/data6/wenchangxi/community_note/src/run_officialschema_nested_cv_fast_2000.py")
        cmd = [
            sys.executable,
            str(script),
            "--run-dir",
            str(out),
            "--folds",
            "5",
            "--inner-folds",
            "4",
        ]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
