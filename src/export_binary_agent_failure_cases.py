from __future__ import annotations

from pathlib import Path

import pandas as pd


RUN_DIR = Path("/data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_20260513")
AGG_DIR = RUN_DIR / "tristate_aggregator_parallel_fast_20260513"
METHOD = "direct_lr_full_C0.1_balanced"
OUT = AGG_DIR / "failure_cases_direct_lr_full_C0.1_balanced.txt"

LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}


def shorten(text: object, n: int = 700) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def main() -> int:
    pred_path = AGG_DIR / "top_method_predictions.csv"
    notes_path = RUN_DIR / "pilot_notes.csv"
    votes_path = RUN_DIR / "agent_votes.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    preds = pd.read_csv(pred_path, dtype={"noteId": str})
    notes = pd.read_csv(notes_path, dtype={"noteId": str, "tweetId": str}, low_memory=False)
    votes = pd.read_csv(votes_path, dtype={"noteId": str}, low_memory=False)
    if METHOD not in preds.columns:
        raise ValueError(f"{METHOD} not found in {pred_path}; columns={preds.columns.tolist()}")

    preds = preds[["noteId", "true_label_3way", "true_label_text", METHOD]].copy()
    preds["pred_label_3way"] = pd.to_numeric(preds[METHOD], errors="coerce").astype(int)
    preds["pred_label_text"] = preds["pred_label_3way"].map(LABEL)
    failed = preds[preds["pred_label_3way"] != preds["true_label_3way"]].copy()

    votes["predicted_rating_score"] = pd.to_numeric(votes["predicted_rating_score"], errors="coerce")
    votes["confidence"] = pd.to_numeric(votes.get("confidence", 0), errors="coerce")
    votes["changes_reader_understanding"] = pd.to_numeric(
        votes.get("changes_reader_understanding", 0), errors="coerce"
    )
    summary = votes.groupby("noteId").agg(
        agent_h=("parsed_rating", lambda s: int((s == "HELPFUL").sum())),
        agent_nh=("parsed_rating", lambda s: int((s == "NOT_HELPFUL").sum())),
        agent_unknown=("parsed_rating", lambda s: int((s == "UNKNOWN").sum())),
        mean_score=("predicted_rating_score", "mean"),
        mean_conf=("confidence", "mean"),
        mean_understand=("changes_reader_understanding", "mean"),
        helpful_context=("helpfulImportantContext", "mean"),
        helpful_claim=("helpfulAddressesClaim", "mean"),
        helpful_sources=("helpfulGoodSources", "mean"),
        nh_missing_points=("notHelpfulMissingKeyPoints", "mean"),
        nh_sources=("notHelpfulSourcesMissingOrUnreliable", "mean"),
        nh_irrelevant_sources=("notHelpfulIrrelevantSources", "mean"),
        nh_note_not_needed=("notHelpfulNoteNotNeeded", "mean"),
        nh_opinion=("notHelpfulOpinionSpeculation", "mean"),
    ).reset_index()

    note_drop = [c for c in ["true_label_3way", "true_label_text"] if c in notes.columns]
    notes_for_merge = notes.drop(columns=note_drop)
    merged = failed.merge(notes_for_merge, on="noteId", how="left").merge(summary, on="noteId", how="left")
    merged["error_type"] = merged["true_label_text"].astype(str) + " -> " + merged["pred_label_text"].astype(str)
    type_counts = merged["error_type"].value_counts().sort_index()

    # Pick representative cases from every confusion cell: high-confidence wrong ones first,
    # then cases with close vote margins, because both are diagnostically useful.
    selected = []
    for err, group in merged.groupby("error_type", sort=True):
        g = group.copy()
        g["vote_margin"] = (g["agent_h"] - g["agent_nh"]).abs()
        g = g.sort_values(["vote_margin", "mean_conf"], ascending=[False, False])
        selected.append(g.head(2))
    cases = pd.concat(selected, ignore_index=True) if selected else merged.head(0)

    lines: list[str] = []
    lines.append(f"Method: {METHOD}")
    lines.append(f"Total failures: {len(merged)} / {len(preds)}")
    lines.append("")
    lines.append("Failure counts:")
    for err, cnt in type_counts.items():
        lines.append(f"- {err}: {cnt}")
    lines.append("")
    lines.append("Representative failure cases:")

    for i, row in cases.iterrows():
        note_id = str(row["noteId"])
        rv = votes[votes["noteId"] == note_id].copy()
        rv = rv.sort_values(["parsed_rating", "confidence"], ascending=[True, False])
        sample_rationales = []
        for _, r in rv.head(4).iterrows():
            sample_rationales.append(
                f"{r.get('agent_id','?')}={r.get('parsed_rating','?')} "
                f"conf={r.get('confidence','?')} reason={shorten(r.get('rationale',''), 170)}"
            )

        post_text = row.get("post_text", row.get("text", ""))
        note_text = row.get("note_text", row.get("summary", ""))
        lines.append("")
        lines.append("=" * 88)
        lines.append(f"Case {i + 1}: {row['error_type']} | noteId={note_id} tweetId={row.get('tweetId','')}")
        lines.append(
            "Agent summary: "
            f"H={int(row['agent_h'])}, NH={int(row['agent_nh'])}, unknown={int(row['agent_unknown'])}, "
            f"mean_score={row['mean_score']:.3f}, mean_conf={row['mean_conf']:.1f}, "
            f"mean_understanding={row['mean_understand']:.1f}"
        )
        lines.append(
            "Reason rates: "
            f"H_context={row['helpful_context']:.2f}, H_claim={row['helpful_claim']:.2f}, "
            f"H_sources={row['helpful_sources']:.2f}, NH_missing={row['nh_missing_points']:.2f}, "
            f"NH_sources={row['nh_sources']:.2f}, NH_irrelevant_sources={row['nh_irrelevant_sources']:.2f}, "
            f"NH_note_not_needed={row['nh_note_not_needed']:.2f}, NH_opinion={row['nh_opinion']:.2f}"
        )
        lines.append(f"POST: {shorten(post_text, 850)}")
        lines.append(f"NOTE: {shorten(note_text, 850)}")
        lines.append("Sample agent rationales:")
        for s in sample_rationales:
            lines.append(f"- {s}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:90]))
    print(f"\nSaved full failure cases to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
