from __future__ import annotations

from pathlib import Path

import pandas as pd


RUN_DIR = Path("/data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_20260513")
AGG_DIR = RUN_DIR / "tristate_aggregator_parallel_fast_20260513"
METHOD = "direct_lr_full_C0.1_balanced"
OUT_CSV = AGG_DIR / "cross_direction_failures_h_to_nh_nh_to_h.csv"
OUT_TXT = AGG_DIR / "cross_direction_failures_h_to_nh_nh_to_h.txt"

LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}


def shorten(value: object, n: int = 420) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def reason_rates(votes: pd.DataFrame) -> dict[str, float]:
    fields = [
        "helpfulClear",
        "helpfulGoodSources",
        "helpfulAddressesClaim",
        "helpfulImportantContext",
        "helpfulUnbiasedLanguage",
        "notHelpfulIncorrect",
        "notHelpfulSourcesMissingOrUnreliable",
        "notHelpfulMissingKeyPoints",
        "notHelpfulHardToUnderstand",
        "notHelpfulArgumentativeOrBiased",
        "notHelpfulIrrelevantSources",
        "notHelpfulOpinionSpeculation",
        "notHelpfulNoteNotNeeded",
    ]
    out = {}
    for field in fields:
        out[field] = float(pd.to_numeric(votes.get(field, 0), errors="coerce").mean())
    return out


def main() -> int:
    preds = pd.read_csv(AGG_DIR / "top_method_predictions.csv", dtype={"noteId": str})
    notes = pd.read_csv(RUN_DIR / "pilot_notes.csv", dtype={"noteId": str, "tweetId": str}, low_memory=False)
    votes = pd.read_csv(RUN_DIR / "agent_votes.csv", dtype={"noteId": str}, low_memory=False)
    preds["pred_label_3way"] = pd.to_numeric(preds[METHOD], errors="coerce").astype(int)
    preds["pred_label_text"] = preds["pred_label_3way"].map(LABEL)

    target = preds[
        ((preds["true_label_3way"] == 2) & (preds["pred_label_3way"] == 0))
        | ((preds["true_label_3way"] == 0) & (preds["pred_label_3way"] == 2))
    ].copy()
    target["error_type"] = target["true_label_text"].astype(str) + " -> " + target["pred_label_text"].astype(str)

    note_drop = [c for c in ["true_label_3way", "true_label_text"] if c in notes.columns]
    merged = target.merge(notes.drop(columns=note_drop), on="noteId", how="left")

    rows = []
    lines = []
    lines.append(f"Method: {METHOD}")
    lines.append(f"Cross-direction failures: {len(merged)}")
    lines.append("")
    for error_type, group in merged.groupby("error_type", sort=True):
        lines.append("=" * 100)
        lines.append(f"{error_type}: {len(group)} cases")
        lines.append("=" * 100)
        for _, row in group.iterrows():
            note_id = str(row["noteId"])
            g = votes[votes["noteId"] == note_id].copy()
            g["confidence"] = pd.to_numeric(g.get("confidence", 0), errors="coerce")
            g["changes_reader_understanding"] = pd.to_numeric(g.get("changes_reader_understanding", 0), errors="coerce")
            h = int((g["parsed_rating"] == "HELPFUL").sum())
            nh = int((g["parsed_rating"] == "NOT_HELPFUL").sum())
            unknown = int((g["parsed_rating"] == "UNKNOWN").sum())
            rr = reason_rates(g)
            post = row.get("post_text", row.get("text", ""))
            note = row.get("note_text", row.get("summary", ""))

            if error_type.startswith("HELPFUL"):
                rationale_pool = g[g["parsed_rating"] == "NOT_HELPFUL"].sort_values("confidence", ascending=False)
            else:
                rationale_pool = g[g["parsed_rating"] == "HELPFUL"].sort_values("confidence", ascending=False)
            if rationale_pool.empty:
                rationale_pool = g.sort_values("confidence", ascending=False)
            rats = []
            for _, r in rationale_pool.head(4).iterrows():
                rats.append(
                    f"{r.get('agent_id')}={r.get('parsed_rating')} conf={r.get('confidence')} "
                    f"reason={shorten(r.get('rationale', ''), 160)}"
                )

            compact = {
                "noteId": note_id,
                "tweetId": row.get("tweetId", ""),
                "error_type": error_type,
                "agent_H": h,
                "agent_NH": nh,
                "agent_UNKNOWN": unknown,
                "mean_confidence": float(g["confidence"].mean()),
                "mean_understanding": float(g["changes_reader_understanding"].mean()),
                "helpfulImportantContext_rate": rr["helpfulImportantContext"],
                "helpfulAddressesClaim_rate": rr["helpfulAddressesClaim"],
                "helpfulGoodSources_rate": rr["helpfulGoodSources"],
                "notHelpfulMissingKeyPoints_rate": rr["notHelpfulMissingKeyPoints"],
                "notHelpfulSourcesMissingOrUnreliable_rate": rr["notHelpfulSourcesMissingOrUnreliable"],
                "notHelpfulIrrelevantSources_rate": rr["notHelpfulIrrelevantSources"],
                "notHelpfulNoteNotNeeded_rate": rr["notHelpfulNoteNotNeeded"],
                "post": shorten(post, 1000),
                "note": shorten(note, 1000),
                "sample_agent_rationales": " || ".join(rats),
            }
            rows.append(compact)

            lines.append("")
            lines.append(f"noteId={note_id} tweetId={row.get('tweetId', '')}")
            lines.append(
                f"Agents: H={h}, NH={nh}, UNKNOWN={unknown}, "
                f"mean_conf={compact['mean_confidence']:.1f}, "
                f"mean_understanding={compact['mean_understanding']:.1f}"
            )
            lines.append(
                "Reason rates: "
                f"H_claim={rr['helpfulAddressesClaim']:.2f}, "
                f"H_context={rr['helpfulImportantContext']:.2f}, "
                f"H_sources={rr['helpfulGoodSources']:.2f}, "
                f"NH_missing={rr['notHelpfulMissingKeyPoints']:.2f}, "
                f"NH_sources={rr['notHelpfulSourcesMissingOrUnreliable']:.2f}, "
                f"NH_irrelevant={rr['notHelpfulIrrelevantSources']:.2f}, "
                f"NH_not_needed={rr['notHelpfulNoteNotNeeded']:.2f}"
            )
            lines.append(f"POST: {shorten(post, 900)}")
            lines.append(f"NOTE: {shorten(note, 900)}")
            lines.append("Agent rationales:")
            for rat in rats:
                lines.append(f"- {rat}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(out[["error_type", "noteId", "agent_H", "agent_NH", "mean_confidence", "mean_understanding", "post", "note"]].to_string(index=False, max_colwidth=170))
    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Saved TXT: {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
