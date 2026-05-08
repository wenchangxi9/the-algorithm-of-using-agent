from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


THREAD_LOCAL = threading.local()
TARGET_STATUSES = {
    "CURRENTLY_RATED_HELPFUL": 1,
    "CURRENTLY_RATED_NOT_HELPFUL": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 72-agent LLM persona simulation for Community Notes helpfulness."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/eval_pool_258/eval_claim_pool_258.csv"),
        help="CSV with noteId, currentStatus, true_label, NoteText, and XText.",
    )
    parser.add_argument(
        "--ratings-cache-csv",
        type=Path,
        default=Path("data/eval_pool_258/ratings_cache_258.csv"),
        help="Optional cached hard ratings used to compute human helpful-share references.",
    )
    parser.add_argument(
        "--persona-labels-file",
        type=Path,
        default=Path("artifacts/agent_variants/mf_continuous_n024/cluster_personas.csv"),
    )
    parser.add_argument(
        "--persona-summary-file",
        type=Path,
        default=Path("artifacts/agent_variants/mf_continuous_n024/cluster_summary.csv"),
    )
    parser.add_argument(
        "--agent-roster-file",
        type=Path,
        default=Path("artifacts/agent_variants/mf_continuous_n024/agent_roster.csv"),
    )
    parser.add_argument(
        "--agent-id-list-file",
        type=Path,
        default=None,
        help="Optional text file with one agent_id per line. If provided, only those agents are run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/llm_runs/mf_continuous_n024_gpt54nano_20260507_run1"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-nano",
        help="Model name served through the configured OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Same-cluster agents share the same prompt, so temperature drives divergence.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Maximum generation length per agent vote.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="How many LLM calls to run in parallel.",
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing agent_votes.csv if present.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Persist partial outputs every N completed calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per failed agent call.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=3.0,
        help="Base backoff before retrying failed requests.",
    )
    return parser.parse_args()


def load_dataset(path: Path | str, max_notes: int | None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path.resolve(), low_memory=False)
    required = {"noteId", "NoteText", "XText"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["noteId"] = df["noteId"].astype(str)
    if "currentStatus" in df.columns:
        df["currentStatus"] = df["currentStatus"].astype(str)
        df = df[df["currentStatus"].isin(TARGET_STATUSES)].copy()
        df["true_label"] = df["currentStatus"].map(TARGET_STATUSES).astype(int)
    elif "true_label" in df.columns:
        df["true_label"] = df["true_label"].astype(int)
        reverse_map = {v: k for k, v in TARGET_STATUSES.items()}
        df["currentStatus"] = df["true_label"].map(reverse_map)
    else:
        raise ValueError(f"{path} must contain currentStatus or true_label")

    df["NoteText"] = df["NoteText"].fillna("").astype(str)
    df["XText"] = df["XText"].fillna("").astype(str)
    df = df[df["NoteText"].str.strip().ne("") & df["XText"].str.strip().ne("")].copy()
    df = df.drop_duplicates(subset=["noteId"]).reset_index(drop=True)
    if max_notes is not None:
        df = df.head(max_notes).copy()
    return df


def load_human_reference(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["noteId", "human_helpful_share", "human_hard_ratings", "human_majority_label"])
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["noteId", "human_helpful_share", "human_hard_ratings", "human_majority_label"])

    ratings = pd.read_csv(
        path.resolve(),
        usecols=["noteId", "hard_label"],
        dtype={"noteId": "string", "hard_label": "int64"},
        low_memory=False,
    )
    ratings["noteId"] = ratings["noteId"].astype(str)
    grouped = ratings.groupby("noteId", as_index=False).agg(
        human_helpful_share=("hard_label", "mean"),
        human_hard_ratings=("hard_label", "size"),
    )
    grouped["human_majority_label"] = (grouped["human_helpful_share"] >= 0.5).astype(int)
    return grouped


def relative_band(value: float, series: pd.Series) -> str:
    rank = float((series <= value).mean())
    if rank >= 0.92:
        return "极高"
    if rank >= 0.75:
        return "较高"
    if rank >= 0.42:
        return "中等"
    if rank >= 0.17:
        return "较低"
    return "极低"


def helpful_tendency_label(share_helpful: float, share_not_helpful: float) -> str:
    margin = share_helpful - share_not_helpful
    if margin >= 0.35:
        return "明显偏向判 Helpful"
    if margin >= 0.15:
        return "温和偏向判 Helpful"
    if margin <= -0.25:
        return "明显偏向判 Not Helpful"
    if margin <= -0.1:
        return "温和偏向判 Not Helpful"
    return "整体接近均衡"


def recent_shift_label(help_shift: float, not_shift: float) -> str:
    if help_shift >= 0.08:
        return "最近明显更愿意给 Helpful"
    if help_shift <= -0.08:
        return "最近明显更不愿意给 Helpful"
    if not_shift >= 0.08:
        return "最近明显更容易否决 note"
    if not_shift <= -0.08:
        return "最近否决倾向有所下降"
    return "最近判断倾向基本稳定"


def authoring_label(crh: float, crnh: float, notes_authored: float) -> str:
    if notes_authored < 1:
        return "几乎不写 note，更像纯评分者"
    if crh >= 0.2:
        return "自己写 note 的成功率较高"
    if crnh >= 0.15:
        return "会积极写 note，但产出里被判不够有帮助的比例也偏高"
    return "会写一些 note，但不是顶级高产作者"


def persona_bias_instruction(
    share_helpful: float,
    share_not_helpful: float,
    strict_band: str,
    evidence_band: str,
) -> str:
    margin = share_helpful - share_not_helpful
    bias_parts: list[str] = []

    if strict_band in {"极高", "较高"}:
        bias_parts.append("你对边界 case 的门槛偏高，但不要因为 note 不够完美或不够全面，就自动否决它。")
    else:
        bias_parts.append("你愿意接受不完美但确实能帮助读者理解原帖的 note。")

    if margin >= 0.15:
        bias_parts.append("在模棱两可的情况下，你会比平均用户更愿意放行有实际帮助的 note。")
    elif margin <= -0.10:
        bias_parts.append("在模棱两可的情况下，你会比平均用户更谨慎，更容易要求 note 证明自己的价值。")
    else:
        bias_parts.append("在模棱两可的情况下，你的判断大体接近均衡。")

    if evidence_band in {"极高", "较高"}:
        bias_parts.append("来源质量、核验力度、是否直接改善读者认知，会明显影响你的判断。")
    else:
        bias_parts.append("你不会苛求学术级证据，但仍要求 note 至少能实质改善读者的理解。")

    return "".join(bias_parts)


def build_persona_prompt(row: pd.Series, summary_df: pd.DataFrame) -> str:
    helpful_label = helpful_tendency_label(float(row["share_helpful"]), float(row["share_not_helpful"]))
    recent_label = recent_shift_label(float(row["recent_helpful_shift"]), float(row["recent_not_helpful_shift"]))
    author_label = authoring_label(
        float(row["share_authored_crh"]),
        float(row["share_authored_crnh"]),
        float(row["notes_authored"]),
    )

    evidence_band = relative_band(float(row["evidence_focus_rate"]), summary_df["evidence_focus_rate"])
    strict_band = relative_band(float(row["strict_rejection_rate"]), summary_df["strict_rejection_rate"])
    civility_band = relative_band(float(row["civility_rejection_rate"]), summary_df["civility_rejection_rate"])
    redundancy_band = relative_band(float(row["redundancy_rejection_rate"]), summary_df["redundancy_rejection_rate"])
    burst_band = relative_band(float(row["activity_burstiness"]), summary_df["activity_burstiness"])
    recent_band = relative_band(float(row["recent_90d_share"]), summary_df["recent_90d_share"])
    note_len_band = relative_band(float(row["avg_summary_char_len"]), summary_df["avg_summary_char_len"])
    bias_instruction = persona_bias_instruction(
        float(row["share_helpful"]),
        float(row["share_not_helpful"]),
        strict_band,
        evidence_band,
    )

    prompt = f"""你现在扮演一名真实的 X Community Notes 评分者，而不是一个通用 AI 助手。

你的固定身份是 Community Notes 用户簇 #{int(row["cluster"])}：{row["persona_name"]}。
你必须稳定模仿这一类用户的判断习惯，不要为了显得客观而退回成“平均用户”。

这类用户的历史行为画像如下：
- 基本风格：{row["activity_label"]}评分，{row["author_label"]}，{row["stance_label"]}，{row["style_label"]}，{row["volatility_label"]}。
- 历史 Helpful / Not Helpful 比例：{float(row["share_helpful"]) * 100:.1f}% / {float(row["share_not_helpful"]) * 100:.1f}%。整体上这类人{helpful_label}。
- 历史活跃度：平均给出 {float(row["avg_ratings_given"]):.1f} 次评分；近 90 天活跃占比 {float(row["recent_90d_share"]) * 100:.1f}%，属于{recent_band}近期活跃；行为爆发性属于{burst_band}。
- 证据敏感度：对“有没有来源、有没有直接回应 claim、有没有补足关键上下文”的看重程度属于{evidence_band}。
- 否决强度：对“证据不足、逻辑跳跃、没有必要、表述武断”的拒绝倾向属于{strict_band}。
- 语气/礼貌敏感度：属于{civility_band}。
- 冗余/没必要敏感度：属于{redundancy_band}。
- 作者侧画像：平均写 note {float(row["notes_authored"]):.1f} 条，{author_label}。
- 已写 note 结果：被判 CRH 比例 {float(row["share_authored_crh"]) * 100:.1f}%，被判 CRNH 比例 {float(row["share_authored_crnh"]) * 100:.1f}%。
- 常见 note 长度偏好：历史上写出的 note 长度属于{note_len_band}。
- 近期漂移：{recent_label}。
- 这类人在边界案例中的自然偏好：{bias_instruction}

你在评分时要遵循这类人的真实偏好：
1. 你评的是 NOTE 本身是否有帮助，不是单纯评原帖真伪，也不是评你政治上赞不赞同。
2. 你会同时看 POST 和 NOTE，判断 note 是否真正改善了普通读者对原帖的理解，而不是机械要求 note 必须逐字逐句复述原帖。
3. 你尤其要判断三件事：
   - 这条 note 是否抓住了原帖的关键误导点，而不是只修边角。
   - 这条 note 是否足以明显改变普通读者的理解。
   - 这条 note 是否真的“有必要成为 Community Note”，而不是虽然部分正确但帮助很小。
4. 只要 note 能实质提升理解，就可以判 HELPFUL。这包括但不限于：纠正身份、时间、地点、数量、来源、图片/视频内容、是否为深度伪造、是否断章取义、是否遗漏关键背景、是否推翻原帖依赖的前提。
5. 但不要把“相关”误当成“有帮助”。如果 note 只是在补充边缘细节、抓字眼、做很小的修正、没有改变读者对原帖核心含义的理解，通常不应判 HELPFUL。
6. 如果 note 通过补上下文、纠正前提、澄清素材来源或指出关键缺失信息，让原帖的核心含义发生明显变化，它即使不是逐字回应，也可以是 HELPFUL。
7. 以下情况更应判 NOT_HELPFUL：note 跑题；只是情绪化反驳；来源弱到不足以支撑结论；遗漏关键点；自身也不准确；只是抠细枝末节而不改变读者理解；重复常识；只表达立场却没有增加有效信息；或者虽然部分为真但并不值得成为一条 Community Note。
8. 要区分“还不够完美”和“真的没有帮助”。前者有时仍应判 HELPFUL，后者才应判 NOT_HELPFUL。
9. 当你拿不准时，不要退回成“平均用户”，而要在上述标准下按这个簇的真实倾向做更自然的判断。

输出要求：
- 只输出 JSON，不要输出 markdown，不要解释规则。
- JSON 格式固定为：
  {{"rating":"HELPFUL" 或 "NOT_HELPFUL","confidence":0到100的整数,"addresses_core_claim":0到100的整数,"changes_reader_understanding":0到100的整数,"note_needed":0到100的整数,"evidence_strength":0到100的整数,"misses_key_points":"YES 或 NO","too_minor_or_tangential":"YES 或 NO","rationale":"不超过35个词的简短理由"}}
"""
    return prompt


def build_persona_library(persona_labels_path: Path | str, persona_summary_path: Path | str) -> pd.DataFrame:
    persona_labels_path = Path(persona_labels_path)
    persona_summary_path = Path(persona_summary_path)
    labels = pd.read_csv(persona_labels_path.resolve(), low_memory=False)
    summary = pd.read_csv(persona_summary_path.resolve(), low_memory=False)
    merged = labels.merge(summary, on="cluster", how="inner", validate="one_to_one")
    merged = merged.sort_values("cluster").reset_index(drop=True)
    if "system_prompt" in merged.columns:
        merged["system_prompt"] = merged["system_prompt"].fillna("").astype(str)
        missing_prompt = merged["system_prompt"].str.strip().eq("")
        if missing_prompt.any():
            merged.loc[missing_prompt, "system_prompt"] = merged.loc[missing_prompt].apply(
                lambda row: build_persona_prompt(row, merged),
                axis=1,
            )
    elif "prompt" in merged.columns:
        merged["system_prompt"] = merged["prompt"].fillna("").astype(str)
        missing_prompt = merged["system_prompt"].str.strip().eq("")
        if missing_prompt.any():
            merged.loc[missing_prompt, "system_prompt"] = merged.loc[missing_prompt].apply(
                lambda row: build_persona_prompt(row, merged),
                axis=1,
            )
    else:
        merged["system_prompt"] = merged.apply(lambda row: build_persona_prompt(row, merged), axis=1)
    return merged


def expand_agent_roster(roster_path: Path | str, persona_df: pd.DataFrame) -> pd.DataFrame:
    roster_path = Path(roster_path)
    roster = pd.read_csv(roster_path.resolve(), low_memory=False)
    rows: list[dict[str, Any]] = []
    prompt_map = {
        int(row.cluster): {
            "persona_name": str(row.persona_name),
            "system_prompt": str(row.system_prompt),
        }
        for row in persona_df.itertuples()
    }
    for entry in roster.sort_values("cluster").itertuples():
        cluster = int(entry.cluster)
        agent_count = int(entry.agent_count)
        prompt_meta = prompt_map[cluster]
        for idx in range(agent_count):
            rows.append(
                {
                    "agent_id": f"cluster_{cluster:02d}_agent_{idx + 1:02d}",
                    "cluster": cluster,
                    "persona_name": prompt_meta["persona_name"],
                    "system_prompt": prompt_meta["system_prompt"],
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError(f"No agent rows were created from {roster_path}")
    return expanded


def filter_agent_roster(expanded_roster: pd.DataFrame, agent_id_list_file: Path | str | None) -> pd.DataFrame:
    if agent_id_list_file is None:
        return expanded_roster

    agent_id_list_file = Path(agent_id_list_file)
    wanted = [
        line.strip()
        for line in agent_id_list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not wanted:
        raise ValueError(f"{agent_id_list_file} did not contain any agent ids.")

    filtered = expanded_roster[expanded_roster["agent_id"].isin(wanted)].copy()
    missing = sorted(set(wanted) - set(filtered["agent_id"].astype(str)))
    if missing:
        raise ValueError(f"Unknown agent ids in {agent_id_list_file}: {missing[:10]}")
    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No agents matched {agent_id_list_file}")
    return filtered


def build_user_prompt(row: pd.Series) -> str:
    parts = []
    datetime_text = str(row.get("DateTimeUTC", "")).strip()
    if datetime_text.lower() == "nan":
        datetime_text = ""
    if datetime_text:
        parts.append(f"POST ({datetime_text}): {str(row['XText']).strip()}")
    else:
        parts.append(f"POST: {str(row['XText']).strip()}")

    quote_text = str(row.get("quoteText", "")).strip()
    if bool(row.get("uses_quote", False)) and quote_text:
        parts.append(f"QUOTED MATERIAL: {quote_text}")

    parts.append(f"NOTE: {str(row['NoteText']).strip()}")
    parts.append(
        '请只根据上面的 POST 和 NOTE 作答，并严格输出 JSON：{"rating":"HELPFUL 或 NOT_HELPFUL","confidence":整数,"addresses_core_claim":整数,"changes_reader_understanding":整数,"note_needed":整数,"evidence_strength":整数,"misses_key_points":"YES/NO","too_minor_or_tangential":"YES/NO","rationale":"简短理由"}'
    )
    return "\n\n".join(parts)


def get_client() -> OpenAI:
    client = getattr(THREAD_LOCAL, "openai_client", None)
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
        THREAD_LOCAL.openai_client = client
    return client


def extract_json_blob(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            continue
    return {}


def normalize_rating_label(value: Any, raw_text: str = "") -> tuple[str, int]:
    text = str(value or "").strip().upper()
    text = text.replace("-", "_").replace(" ", "_")
    if text in {"NOT_HELPFUL", "NOTHELPFUL"}:
        return "NOT_HELPFUL", 0
    if text == "HELPFUL":
        return "HELPFUL", 1

    fallback = str(raw_text or "").upper()
    if "NOT_HELPFUL" in fallback or re.search(r"\bNOT\s+HELPFUL\b", fallback):
        return "NOT_HELPFUL", 0
    if re.search(r"\bHELPFUL\b", fallback):
        return "HELPFUL", 1
    return "UNKNOWN", -1


def parse_confidence(value: Any) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return -1
    return max(0, min(parsed, 100))


def parse_score_0_100(value: Any) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return -1
    return max(0, min(parsed, 100))


def parse_yes_no(value: Any) -> int:
    text = str(value or "").strip().upper()
    text = text.replace("-", "_").replace(" ", "_")
    if text in {"YES", "Y", "TRUE", "1", "是"}:
        return 1
    if text in {"NO", "N", "FALSE", "0", "否"}:
        return 0
    return -1


def parse_response(text: str) -> dict[str, Any]:
    payload = extract_json_blob(text)
    rating_text, predicted_label = normalize_rating_label(payload.get("rating", ""), text)
    confidence = parse_confidence(payload.get("confidence"))
    addresses_core_claim = parse_score_0_100(payload.get("addresses_core_claim"))
    changes_reader_understanding = parse_score_0_100(payload.get("changes_reader_understanding"))
    note_needed = parse_score_0_100(payload.get("note_needed"))
    evidence_strength = parse_score_0_100(payload.get("evidence_strength"))
    misses_key_points = parse_yes_no(payload.get("misses_key_points"))
    too_minor_or_tangential = parse_yes_no(payload.get("too_minor_or_tangential"))
    rationale = str(payload.get("rationale", "")).strip()
    return {
        "parsed_rating": rating_text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "addresses_core_claim": addresses_core_claim,
        "changes_reader_understanding": changes_reader_understanding,
        "note_needed": note_needed,
        "evidence_strength": evidence_strength,
        "misses_key_points": misses_key_points,
        "too_minor_or_tangential": too_minor_or_tangential,
        "rationale": rationale,
    }


def run_agent_call(
    note_row: pd.Series,
    agent_row: pd.Series,
    args: argparse.Namespace,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(note_row)
    last_error = ""
    raw_completion = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            response = get_client().chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                messages=[
                    {"role": "system", "content": str(agent_row["system_prompt"])},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_completion = response.choices[0].message.content or ""
            last_error = ""
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(args.retry_delay_seconds * attempt)

    parsed = parse_response(raw_completion)
    return {
        "noteId": str(note_row["noteId"]),
        "agent_id": str(agent_row["agent_id"]),
        "cluster": int(agent_row["cluster"]),
        "persona_name": str(agent_row["persona_name"]),
        "raw_completion": raw_completion,
        "parsed_rating": parsed["parsed_rating"],
        "predicted_label": int(parsed["predicted_label"]),
        "confidence": int(parsed["confidence"]),
        "addresses_core_claim": int(parsed["addresses_core_claim"]),
        "changes_reader_understanding": int(parsed["changes_reader_understanding"]),
        "note_needed": int(parsed["note_needed"]),
        "evidence_strength": int(parsed["evidence_strength"]),
        "misses_key_points": int(parsed["misses_key_points"]),
        "too_minor_or_tangential": int(parsed["too_minor_or_tangential"]),
        "rationale": parsed["rationale"],
        "api_error": last_error,
    }


def binary_confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> tuple[int, int, int, int]:
    true_vals = y_true.astype(int)
    pred_vals = y_pred.astype(int)
    tn = int(((true_vals == 0) & (pred_vals == 0)).sum())
    fp = int(((true_vals == 0) & (pred_vals == 1)).sum())
    fn = int(((true_vals == 1) & (pred_vals == 0)).sum())
    tp = int(((true_vals == 1) & (pred_vals == 1)).sum())
    return tn, fp, fn, tp


def balanced_accuracy_from_counts(tn: int, fp: int, fn: int, tp: int) -> float:
    recall_not_helpful = float(tn / (tn + fp)) if (tn + fp) else 0.0
    recall_helpful = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return float((recall_not_helpful + recall_helpful) / 2.0)


def compute_metrics(note_predictions: pd.DataFrame) -> dict[str, Any]:
    valid = note_predictions[note_predictions["llm_pred_label"].isin([0, 1])].copy()
    if valid.empty:
        return {
            "evaluated_notes": 0,
            "coverage": 0.0,
            "accuracy_vs_status": 0.0,
            "balanced_accuracy_vs_status": 0.0,
            "f1_vs_status": 0.0,
            "accuracy_vs_human_majority": 0.0,
            "helpful_share_corr_vs_human": 0.0,
            "helpful_share_mae_vs_human": 0.0,
            "recall_not_helpful": 0.0,
            "recall_helpful": 0.0,
        }

    accuracy = float(accuracy_score(valid["true_label"], valid["llm_pred_label"]))
    f1 = float(f1_score(valid["true_label"], valid["llm_pred_label"], zero_division=0))
    tn, fp, fn, tp = binary_confusion_counts(valid["true_label"], valid["llm_pred_label"])
    balanced = balanced_accuracy_from_counts(tn, fp, fn, tp)

    if valid["human_majority_label"].isin([0, 1]).all():
        human_majority_acc = float(accuracy_score(valid["human_majority_label"], valid["llm_pred_label"]))
    else:
        human_majority_acc = 0.0

    share_corr = 0.0
    share_mae = 0.0
    if valid["human_helpful_share"].notna().sum() >= 2:
        llm_shares = valid["llm_helpful_share"].astype(float)
        human_shares = valid["human_helpful_share"].astype(float)
        if float(llm_shares.std(ddof=0)) > 0.0 and float(human_shares.std(ddof=0)) > 0.0:
            share_corr = float(np.corrcoef(llm_shares, human_shares)[0, 1])
            if math.isnan(share_corr):
                share_corr = 0.0
        share_mae = float(mean_absolute_error(valid["human_helpful_share"], valid["llm_helpful_share"]))

    recall_not_helpful = float(tn / (tn + fp)) if (tn + fp) else 0.0
    recall_helpful = float(tp / (tp + fn)) if (tp + fn) else 0.0

    return {
        "evaluated_notes": int(len(valid)),
        "coverage": float(len(valid) / len(note_predictions)),
        "accuracy_vs_status": accuracy,
        "balanced_accuracy_vs_status": balanced,
        "f1_vs_status": f1,
        "accuracy_vs_human_majority": human_majority_acc,
        "helpful_share_corr_vs_human": share_corr,
        "helpful_share_mae_vs_human": share_mae,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "recall_not_helpful": recall_not_helpful,
        "recall_helpful": recall_helpful,
    }


def aggregate_predictions(
    dataset_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    human_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid_votes = votes_df[votes_df["predicted_label"].isin([0, 1])].copy()

    note_level = valid_votes.groupby("noteId", as_index=False).agg(
        llm_helpful_votes=("predicted_label", "sum"),
        llm_total_votes=("predicted_label", "size"),
        llm_helpful_share=("predicted_label", "mean"),
        llm_mean_confidence=("confidence", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        llm_mean_addresses_core_claim=("addresses_core_claim", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        llm_mean_changes_reader_understanding=("changes_reader_understanding", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        llm_mean_note_needed=("note_needed", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        llm_mean_evidence_strength=("evidence_strength", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        llm_misses_key_points_rate=("misses_key_points", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        llm_too_minor_rate=("too_minor_or_tangential", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
    )
    note_level["llm_pred_label"] = (note_level["llm_helpful_share"] >= 0.5).astype(int)

    cluster_level = valid_votes.groupby(["noteId", "cluster"], as_index=False).agg(
        cluster_helpful_share=("predicted_label", "mean"),
        cluster_votes=("predicted_label", "size"),
        cluster_confidence=("confidence", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        cluster_addresses_core_claim=("addresses_core_claim", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        cluster_changes_reader_understanding=("changes_reader_understanding", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        cluster_note_needed=("note_needed", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        cluster_evidence_strength=("evidence_strength", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        cluster_misses_key_points_rate=("misses_key_points", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
        cluster_too_minor_rate=("too_minor_or_tangential", lambda s: float(pd.Series(s).replace(-1, np.nan).mean())),
    )

    mixed_same_cluster = (
        cluster_level.assign(mixed=lambda df: (df["cluster_helpful_share"] > 0) & (df["cluster_helpful_share"] < 1))
        .groupby("cluster", as_index=False)["mixed"]
        .mean()
        .rename(columns={"mixed": "same_prompt_divergence_rate"})
    )

    cluster_json_rows = []
    for note_id, group in cluster_level.sort_values(["noteId", "cluster"]).groupby("noteId", sort=False):
        cluster_json_rows.append(
            {
                "noteId": note_id,
                "cluster_vote_profile_json": json.dumps(
                    {
                        str(int(row.cluster)): {
                            "helpful_share": round(float(row.cluster_helpful_share), 6),
                            "votes": int(row.cluster_votes),
                            "addresses_core_claim": round(float(row.cluster_addresses_core_claim), 3),
                            "changes_reader_understanding": round(float(row.cluster_changes_reader_understanding), 3),
                            "note_needed": round(float(row.cluster_note_needed), 3),
                            "evidence_strength": round(float(row.cluster_evidence_strength), 3),
                            "misses_key_points_rate": round(float(row.cluster_misses_key_points_rate), 6),
                            "too_minor_rate": round(float(row.cluster_too_minor_rate), 6),
                        }
                        for row in group.itertuples()
                    },
                    ensure_ascii=False,
                ),
            }
        )
    cluster_json = pd.DataFrame(cluster_json_rows)
    if cluster_json.empty:
        cluster_json = pd.DataFrame(columns=["noteId", "cluster_vote_profile_json"])

    note_predictions = dataset_df.merge(note_level, on="noteId", how="left")
    note_predictions = note_predictions.merge(cluster_json, on="noteId", how="left")
    note_predictions = note_predictions.merge(human_reference, on="noteId", how="left")
    note_predictions["human_majority_label"] = note_predictions["human_majority_label"].fillna(note_predictions["true_label"])
    metrics = compute_metrics(note_predictions)
    metrics["same_prompt_divergence_by_cluster"] = {
        str(int(row.cluster)): float(row.same_prompt_divergence_rate) for row in mixed_same_cluster.itertuples()
    }
    metrics["same_prompt_divergence_overall"] = float(mixed_same_cluster["same_prompt_divergence_rate"].mean()) if not mixed_same_cluster.empty else 0.0
    return note_predictions, cluster_level.merge(mixed_same_cluster, on="cluster", how="left"), metrics


def load_existing_predictions(path: Path, note_col: str, pred_col: str, rename_to: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["noteId", rename_to])
    df = pd.read_csv(path.resolve(), low_memory=False)
    if note_col not in df.columns or pred_col not in df.columns:
        return pd.DataFrame(columns=["noteId", rename_to])
    out = df[[note_col, pred_col]].copy()
    out.columns = ["noteId", rename_to]
    out["noteId"] = out["noteId"].astype(str)
    return out.drop_duplicates(subset=["noteId"])


def build_comparison(note_predictions: pd.DataFrame, output_dir: Path) -> None:
    comparison_note_df = note_predictions[["noteId", "true_label", "human_majority_label", "llm_pred_label"]].copy()

    def metric_block(pred_col: str) -> dict[str, float]:
        valid = comparison_note_df[comparison_note_df[pred_col].isin([0, 1])]
        if valid.empty:
            return {
                "accuracy_vs_status": 0.0,
                "balanced_accuracy_vs_status": 0.0,
                "f1_vs_status": 0.0,
                "accuracy_vs_human_majority": 0.0,
            }
        tn, fp, fn, tp = binary_confusion_counts(valid["true_label"], valid[pred_col])
        return {
            "accuracy_vs_status": float(accuracy_score(valid["true_label"], valid[pred_col])),
            "balanced_accuracy_vs_status": balanced_accuracy_from_counts(tn, fp, fn, tp),
            "f1_vs_status": float(f1_score(valid["true_label"], valid[pred_col], zero_division=0)),
            "accuracy_vs_human_majority": float(accuracy_score(valid["human_majority_label"], valid[pred_col])),
        }

    comparison_metrics = {
        "MFContinuous_LLM_majority": metric_block("llm_pred_label"),
    }
    merged_summary = {
        "pool_notes": int(len(comparison_note_df)),
        "metrics": comparison_metrics,
    }
    with (output_dir / "comparison_with_existing_models.json").open("w", encoding="utf-8") as handle:
        json.dump(merged_summary, handle, ensure_ascii=False, indent=2)

    rows = [{"model": model_name, **metrics} for model_name, metrics in comparison_metrics.items()]
    pd.DataFrame(rows).drop_duplicates(subset=["model"], keep="last").to_csv(
        output_dir / "comparison_with_existing_models.csv", index=False
    )

    comparison_note_df.to_csv(output_dir / "note_level_model_agreement.csv", index=False)


def save_outputs(
    votes_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    human_reference: pd.DataFrame,
    persona_df: pd.DataFrame,
    expanded_roster: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    votes_df = votes_df.sort_values(["noteId", "agent_id"]).reset_index(drop=True)
    note_predictions, cluster_votes, metrics = aggregate_predictions(dataset_df, votes_df, human_reference)

    persona_df[["cluster", "persona_name", "system_prompt"]].to_csv(output_dir / "persona_prompts.csv", index=False)
    expanded_roster[["agent_id", "cluster", "persona_name"]].to_csv(
        output_dir / "agent_roster_expanded.csv", index=False
    )
    votes_df.to_csv(output_dir / "agent_votes.csv", index=False)
    cluster_votes.to_csv(output_dir / "cluster_vote_breakdown.csv", index=False)
    note_predictions.to_csv(output_dir / "note_predictions.csv", index=False)

    run_metadata = {
        "dataset_csv": str(args.dataset_csv),
        "ratings_cache_csv": str(args.ratings_cache_csv),
        "notes_total": int(len(dataset_df)),
        "agent_total": int(len(expanded_roster)),
        "vote_calls_total": int(len(dataset_df) * len(expanded_roster)),
        "successful_votes": int(votes_df["predicted_label"].isin([0, 1]).sum()),
        "failed_votes": int((votes_df["predicted_label"] == -1).sum()),
        "model": args.model,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "metrics": metrics,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, ensure_ascii=False, indent=2)

    build_comparison(note_predictions, output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_dataset(args.dataset_csv, args.max_notes)
    human_reference = load_human_reference(args.ratings_cache_csv)
    persona_df = build_persona_library(args.persona_labels_file, args.persona_summary_file)
    expanded_roster = expand_agent_roster(args.agent_roster_file, persona_df)
    expanded_roster = filter_agent_roster(expanded_roster, args.agent_id_list_file)

    votes_path = output_dir / "agent_votes.csv"
    existing_votes_df = pd.DataFrame(
        columns=[
            "noteId",
            "agent_id",
            "cluster",
            "persona_name",
            "raw_completion",
            "parsed_rating",
            "predicted_label",
            "confidence",
            "addresses_core_claim",
            "changes_reader_understanding",
            "note_needed",
            "evidence_strength",
            "misses_key_points",
            "too_minor_or_tangential",
            "rationale",
            "api_error",
        ]
    )
    if args.resume and votes_path.exists():
        existing_votes_df = pd.read_csv(votes_path.resolve(), low_memory=False)
        existing_votes_df["noteId"] = existing_votes_df["noteId"].astype(str)
        existing_votes_df["agent_id"] = existing_votes_df["agent_id"].astype(str)
        existing_votes_df = existing_votes_df[existing_votes_df["predicted_label"].isin([0, 1])].copy()

    processed_keys = {
        (str(row.noteId), str(row.agent_id))
        for row in existing_votes_df.itertuples()
        if int(row.predicted_label) in (0, 1)
    }

    note_lookup = {str(row.noteId): row for _, row in dataset_df.iterrows()}
    jobs: list[tuple[str, pd.Series]] = []
    for _, agent_row in expanded_roster.iterrows():
        for note_id in dataset_df["noteId"].tolist():
            if (str(note_id), str(agent_row["agent_id"])) not in processed_keys:
                jobs.append((str(note_id), agent_row))

    all_votes: list[dict[str, Any]] = existing_votes_df.to_dict(orient="records")
    completed_since_save = 0

    def submit_job(executor: ThreadPoolExecutor, note_id: str, agent_row: pd.Series):
        return executor.submit(run_agent_call, note_lookup[note_id], agent_row, args)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending_jobs = iter(jobs)
        futures: dict[Any, tuple[str, str]] = {}

        for _ in range(min(args.concurrency, len(jobs))):
            try:
                note_id, agent_row = next(pending_jobs)
            except StopIteration:
                break
            future = submit_job(executor, note_id, agent_row)
            futures[future] = (note_id, str(agent_row["agent_id"]))

        total_calls = len(dataset_df) * len(expanded_roster)
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                note_id, agent_id = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive path
                    result = {
                        "noteId": note_id,
                        "agent_id": agent_id,
                        "cluster": -1,
                        "persona_name": "",
                        "raw_completion": "",
                        "parsed_rating": "UNKNOWN",
                        "predicted_label": -1,
                        "confidence": -1,
                        "addresses_core_claim": -1,
                        "changes_reader_understanding": -1,
                        "note_needed": -1,
                        "evidence_strength": -1,
                        "misses_key_points": -1,
                        "too_minor_or_tangential": -1,
                        "rationale": "",
                        "api_error": f"{type(exc).__name__}: {exc}",
                    }
                all_votes.append(result)
                completed_since_save += 1

                if completed_since_save >= args.save_every:
                    save_outputs(
                        votes_df=pd.DataFrame(all_votes),
                        dataset_df=dataset_df,
                        human_reference=human_reference,
                        persona_df=persona_df,
                        expanded_roster=expanded_roster,
                        output_dir=output_dir,
                        args=args,
                    )
                    completed_total = len(all_votes)
                    print(
                        json.dumps(
                            {
                                "progress_votes_saved": completed_total,
                                "vote_calls_total": total_calls,
                                "coverage": round(completed_total / total_calls, 4) if total_calls else 0.0,
                            },
                            ensure_ascii=False,
                        )
                    )
                    completed_since_save = 0

                try:
                    next_note_id, next_agent_row = next(pending_jobs)
                    next_future = submit_job(executor, next_note_id, next_agent_row)
                    futures[next_future] = (next_note_id, str(next_agent_row["agent_id"]))
                except StopIteration:
                    pass

    save_outputs(
        votes_df=pd.DataFrame(all_votes),
        dataset_df=dataset_df,
        human_reference=human_reference,
        persona_df=persona_df,
        expanded_roster=expanded_roster,
        output_dir=output_dir,
        args=args,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "notes_total": int(len(dataset_df)),
                "agent_total": int(len(expanded_roster)),
                "vote_calls_total": int(len(dataset_df) * len(expanded_roster)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
