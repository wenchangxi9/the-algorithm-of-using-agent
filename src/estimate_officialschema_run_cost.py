from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN_DIR = Path("/data6/wenchangxi/community_note/analysis/llm_16agent_rawrating_2000_20260512_officialschema")


def approx_tokens(chars: float) -> float:
    # Conservative English-heavy approximation. Exact billing requires provider token usage logs.
    return chars / 4.0


def main() -> int:
    personas = pd.read_csv(RUN_DIR / "persona_prompts.csv", low_memory=False)
    notes = pd.read_csv(RUN_DIR / "pilot_notes.csv", low_memory=False)
    votes = pd.read_csv(RUN_DIR / "agent_votes.csv", low_memory=False)

    note_col = "note_text" if "note_text" in notes.columns else "summary"
    post_col = "post_text" if "post_text" in notes.columns else "text"
    user_prompts = (
        "POST:\n"
        + notes.get(post_col, pd.Series([""] * len(notes))).fillna("").astype(str)
        + "\n\nCOMMUNITY NOTE:\n"
        + notes.get(note_col, pd.Series([""] * len(notes))).fillna("").astype(str)
        + "\n\nEvaluate only the note's usefulness for the post. Do not infer the official status from metadata."
    )

    n_notes = len(notes)
    n_agents = len(personas)
    n_calls = n_notes * n_agents
    system_chars_mean = personas["system_prompt"].fillna("").astype(str).str.len().mean()
    user_chars_mean = user_prompts.str.len().mean()
    completion_chars_mean = votes["raw_completion"].fillna("").astype(str).str.len().mean()

    input_tokens_per_call = approx_tokens(system_chars_mean + user_chars_mean)
    output_tokens_per_call = approx_tokens(completion_chars_mean)
    input_tokens_total = input_tokens_per_call * n_calls
    output_tokens_total = output_tokens_per_call * n_calls

    prices = {
        "gpt-5.4": {"input": 2.50, "output": 15.00},
        "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    }
    costs = {}
    for model, price in prices.items():
        input_cost = input_tokens_total / 1_000_000 * price["input"]
        output_cost = output_tokens_total / 1_000_000 * price["output"]
        costs[model] = {
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }

    result = {
        "n_notes": n_notes,
        "n_agents": n_agents,
        "n_calls": n_calls,
        "system_chars_mean": system_chars_mean,
        "user_chars_mean": user_chars_mean,
        "completion_chars_mean": completion_chars_mean,
        "approx_input_tokens_per_call": input_tokens_per_call,
        "approx_output_tokens_per_call": output_tokens_per_call,
        "approx_input_tokens_total": input_tokens_total,
        "approx_output_tokens_total": output_tokens_total,
        "costs": costs,
        "note": "Approximate token estimate uses chars/4. Use provider token usage logs for exact billing.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (RUN_DIR / "cost_estimate_officialschema_2000.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
