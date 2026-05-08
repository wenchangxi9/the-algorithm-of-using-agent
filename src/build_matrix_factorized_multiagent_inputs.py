from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


OLD_FEATURE_COLUMNS = [
    "participantId",
    "ratings_given",
    "ratings_helpful",
    "ratings_not_helpful",
    "ratings_somewhat_helpful",
    "helpfulInformative",
    "helpfulClear",
    "helpfulEmpathetic",
    "helpfulGoodSources",
    "helpfulUniqueContext",
    "helpfulAddressesClaim",
    "helpfulImportantContext",
    "helpfulUnbiasedLanguage",
    "notHelpfulIncorrect",
    "notHelpfulSourcesMissingOrUnreliable",
    "notHelpfulOpinionSpeculationOrBias",
    "notHelpfulMissingKeyPoints",
    "notHelpfulOutdated",
    "notHelpfulHardToUnderstand",
    "notHelpfulArgumentativeOrBiased",
    "notHelpfulOffTopic",
    "notHelpfulSpamHarassmentOrAbuse",
    "notHelpfulIrrelevantSources",
    "notHelpfulNoteNotNeeded",
    "first_rating_ts",
    "last_rating_ts",
    "notes_authored",
    "notes_misleading",
    "notes_not_misleading",
    "authored_media_notes",
    "authored_collaborative_notes",
    "first_note_ts",
    "last_note_ts",
    "avg_summary_char_len",
    "authored_status_crh",
    "authored_status_crnh",
    "authored_status_nmr",
    "successfulRatingNeededToEarnIn",
    "timestampOfLastStateChange",
    "timestampOfLastEarnOut",
    "modelingGroup",
    "numberOfTimesEarnedOut",
    "population_CORE",
    "population_EXPANSION",
    "population_EXPANSION_PLUS",
    "state_apiEarnedIn",
    "state_atRisk",
    "state_earnedIn",
    "state_earnedOutAcknowledged",
    "state_newUser",
    "state_removed",
    "rating_active_days",
    "note_active_days",
    "days_since_last_state_change",
    "days_since_last_earn_out",
    "share_helpful",
    "share_not_helpful",
    "share_somewhat_helpful",
    "share_notes_misleading",
    "share_notes_not_misleading",
    "share_authored_media_notes",
    "share_authored_collaborative_notes",
    "share_authored_crh",
    "share_authored_crnh",
    "activity_score",
    "cluster",
    "active_buckets",
    "bucket_mean_ratings",
    "bucket_std_ratings",
    "bucket_max_ratings",
    "recent_90d_ratings",
    "recent_90d_helpful",
    "recent_90d_not_helpful",
    "ratings_per_active_day",
    "notes_per_active_day",
    "author_vs_rater_ratio",
    "activity_burstiness",
    "max_bucket_share",
    "recent_90d_share",
    "recent_90d_helpful_share",
    "recent_90d_not_helpful_share",
    "recent_helpful_shift",
    "recent_not_helpful_shift",
    "evidence_focus_rate",
    "tone_focus_rate",
    "novelty_focus_rate",
    "strict_rejection_rate",
    "civility_rejection_rate",
    "redundancy_rejection_rate",
    "author_success_balance",
    "days_since_last_rating",
    "persona_cluster",
]

NEW_FEATURE_COLUMNS = [
    "participantId",
    "cluster",
    "old_cluster",
    "bw_final_rater_intercept",
    "bw_final_rater_factor_1",
    "bw_pre_rater_intercept",
    "bw_pre_rater_factor_1",
    "bw_rater_agree_ratio",
    "bw_valid_rating_count",
    "bw_successful_rating_count",
    "bw_unsuccessful_rating_count",
    "bw_mean_note_score",
    "bw_crh_ratio",
    "bw_crnh_ratio",
    "bw_crh_crnh_ratio_difference",
    "bw_helpfulness_pass",
]

SUMMARY_MEAN_COLUMNS = [
    "ratings_given",
    "ratings_per_active_day",
    "recent_90d_share",
    "activity_burstiness",
    "max_bucket_share",
    "days_since_last_rating",
    "share_helpful",
    "share_not_helpful",
    "share_somewhat_helpful",
    "recent_helpful_shift",
    "recent_not_helpful_shift",
    "evidence_focus_rate",
    "tone_focus_rate",
    "novelty_focus_rate",
    "strict_rejection_rate",
    "civility_rejection_rate",
    "redundancy_rejection_rate",
    "notes_authored",
    "notes_per_active_day",
    "author_vs_rater_ratio",
    "share_notes_misleading",
    "share_notes_not_misleading",
    "share_authored_media_notes",
    "share_authored_collaborative_notes",
    "share_authored_crh",
    "share_authored_crnh",
    "author_success_balance",
    "avg_summary_char_len",
    "successfulRatingNeededToEarnIn",
    "modelingGroup",
    "numberOfTimesEarnedOut",
    "days_since_last_state_change",
    "days_since_last_earn_out",
    "population_CORE",
    "population_EXPANSION",
    "population_EXPANSION_PLUS",
    "state_apiEarnedIn",
    "state_atRisk",
    "state_earnedIn",
    "state_earnedOutAcknowledged",
    "state_newUser",
    "state_removed",
    "bw_final_rater_intercept",
    "bw_final_rater_factor_1",
    "bw_pre_rater_intercept",
    "bw_pre_rater_factor_1",
    "bw_rater_agree_ratio",
    "bw_valid_rating_count",
    "bw_successful_rating_count",
    "bw_unsuccessful_rating_count",
    "bw_mean_note_score",
    "bw_crh_ratio",
    "bw_crnh_ratio",
    "bw_crh_crnh_ratio_difference",
    "bw_helpfulness_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build persona prompts and weighted 72-agent roster from matrix-factorized Community Notes clusters."
    )
    parser.add_argument(
        "--old-features",
        type=Path,
        default=Path("data/base_user_features/user_features_with_behavior_features.csv"),
    )
    parser.add_argument(
        "--new-features",
        type=Path,
        default=Path("artifacts/mf_clustering/user_features_with_mf_clusters.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/persona_inputs"),
    )
    parser.add_argument("--total-agents", type=int, default=72)
    parser.add_argument(
        "--allow-zero-agent-clusters",
        action="store_true",
        help="If set, extremely small clusters may receive zero agents.",
    )
    return parser.parse_args()


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
    parts: list[str] = []
    if strict_band in {"极高", "较高"}:
        parts.append("你对边界 case 的门槛偏高，但不要因为 note 不够完美就自动否决它。")
    else:
        parts.append("你愿意接受不完美但确实能帮助读者理解原帖的 note。")
    if margin >= 0.15:
        parts.append("在模棱两可的情况下，你会比平均用户更愿意放行有实际帮助的 note。")
    elif margin <= -0.10:
        parts.append("在模棱两可的情况下，你会比平均用户更谨慎，更容易要求 note 证明自己的价值。")
    else:
        parts.append("在模棱两可的情况下，你的判断大体接近均衡。")
    if evidence_band in {"极高", "较高"}:
        parts.append("来源质量、核验力度、是否直接改善读者认知，会明显影响你的判断。")
    else:
        parts.append("你不会苛求学术级证据，但仍要求 note 至少能实质改善读者的理解。")
    return "".join(parts)


def allocate_agents(users: pd.Series, total_agents: int, allow_zero: bool) -> pd.Series:
    weights = users.astype(float) / float(users.sum())
    if allow_zero:
        base = np.zeros(len(users), dtype=int)
        remaining = total_agents
    else:
        if len(users) > total_agents:
            raise ValueError("total_agents must be >= number of clusters when zero-agent clusters are disallowed.")
        base = np.ones(len(users), dtype=int)
        remaining = total_agents - len(users)

    raw = weights.to_numpy() * remaining
    allocation = base + np.floor(raw).astype(int)
    shortfall = int(total_agents - allocation.sum())
    remainders = raw - np.floor(raw)
    order = sorted(
        range(len(users)),
        key=lambda idx: (-remainders[idx], -float(users.iloc[idx]), int(users.index[idx])),
    )
    for idx in order[:shortfall]:
        allocation[idx] += 1
    return pd.Series(allocation, index=users.index, dtype="int64")


def build_labels(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    note_score_q85 = summary["bw_mean_note_score"].quantile(0.85)
    authored_q75 = summary["notes_authored"].quantile(0.75)
    agree_q75 = summary["bw_rater_agree_ratio"].quantile(0.75)

    for row in summary.itertuples(index=False):
        margin = float(row.share_helpful) - float(row.share_not_helpful)
        shift_mag = abs(float(row.recent_helpful_shift)) + abs(float(row.recent_not_helpful_shift))

        if float(row.bw_mean_note_score) >= note_score_q85:
            persona_name = "高质量作者型贡献者"
        elif float(row.notes_authored) >= authored_q75:
            persona_name = "资深写评混合型贡献者"
        elif float(row.bw_rater_agree_ratio) >= agree_q75 and margin <= -0.05:
            persona_name = "严格共识型评分者"
        elif float(row.bw_rater_agree_ratio) >= agree_q75 and margin >= 0.05:
            persona_name = "主流 Helpful 共识型评分者"
        elif margin >= 0.15 and float(row.bw_final_rater_factor_1) >= 0.1:
            persona_name = "正向视角的宽松 Helpful 评分者"
        elif margin >= 0.15 and float(row.bw_final_rater_factor_1) <= -0.1:
            persona_name = "反向视角的宽松 Helpful 评分者"
        else:
            persona_name = "平衡型评分者"

        ratings_band = relative_band(float(row.avg_ratings_given), summary["avg_ratings_given"])
        activity_label = {
            "极高": "高频",
            "较高": "高频",
            "中等": "中频",
            "较低": "低频",
            "极低": "低频",
        }[ratings_band]

        if float(row.notes_authored) < 1:
            author_label_short = "几乎不写 note"
        elif float(row.notes_authored) < 4:
            author_label_short = "少写 note"
        elif float(row.notes_authored) < 10:
            author_label_short = "常写 note"
        else:
            author_label_short = "高产 note 作者"

        if float(row.bw_mean_note_score) >= note_score_q85:
            style_label = "作者视角强，也看重 note 是否值得发布"
        elif relative_band(float(row.evidence_focus_rate), summary["evidence_focus_rate"]) in {"极高", "较高"} and relative_band(
            float(row.strict_rejection_rate), summary["strict_rejection_rate"]
        ) in {"极高", "较高"}:
            style_label = "证据导向且门槛较高"
        elif relative_band(float(row.evidence_focus_rate), summary["evidence_focus_rate"]) in {"极高", "较高"}:
            style_label = "证据导向"
        else:
            style_label = "平衡判断"

        if shift_mag < 0.05 and relative_band(float(row.activity_burstiness), summary["activity_burstiness"]) in {"极高", "较高"}:
            volatility_label = "高波动，近期稳定"
        elif shift_mag < 0.05:
            volatility_label = "近期稳定"
        elif float(row.recent_helpful_shift) >= 0.08:
            volatility_label = "近期更偏 Helpful"
        elif float(row.recent_not_helpful_shift) >= 0.08:
            volatility_label = "近期更偏否决"
        else:
            volatility_label = "近期有轻微漂移"

        if margin >= 0.12:
            stance_label = "更容易给 Helpful"
        elif margin <= -0.08:
            stance_label = "更容易给 Not Helpful"
        else:
            stance_label = "Helpful / Not Helpful 相对均衡"

        rows.append(
            {
                "cluster": int(row.cluster),
                "persona_name": persona_name,
                "activity_label": activity_label,
                "author_label": author_label_short,
                "stance_label": stance_label,
                "style_label": style_label,
                "volatility_label": volatility_label,
            }
        )

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def build_system_prompt(row: pd.Series, summary: pd.DataFrame) -> str:
    evidence_band = relative_band(float(row["evidence_focus_rate"]), summary["evidence_focus_rate"])
    strict_band = relative_band(float(row["strict_rejection_rate"]), summary["strict_rejection_rate"])
    redundancy_band = relative_band(float(row["redundancy_rejection_rate"]), summary["redundancy_rejection_rate"])
    recent_band = relative_band(float(row["recent_90d_share"]), summary["recent_90d_share"])
    note_len_band = relative_band(float(row["avg_summary_char_len"]), summary["avg_summary_char_len"])
    agree_band = relative_band(float(row["bw_rater_agree_ratio"]), summary["bw_rater_agree_ratio"])
    note_score_band = relative_band(float(row["bw_mean_note_score"]), summary["bw_mean_note_score"])
    factor = float(row["bw_final_rater_factor_1"])
    if factor >= 0.1:
        latent_label = "位于潜在判断视角轴的正向一侧"
    elif factor <= -0.1:
        latent_label = "位于潜在判断视角轴的负向一侧"
    else:
        latent_label = "位于潜在判断视角轴的中间区域"

    bias_instruction = persona_bias_instruction(
        float(row["share_helpful"]),
        float(row["share_not_helpful"]),
        strict_band,
        evidence_band,
    )
    helpful_label = helpful_tendency_label(float(row["share_helpful"]), float(row["share_not_helpful"]))
    recent_label = recent_shift_label(float(row["recent_helpful_shift"]), float(row["recent_not_helpful_shift"]))
    author_label = authoring_label(
        float(row["share_authored_crh"]),
        float(row["share_authored_crnh"]),
        float(row["notes_authored"]),
    )

    extra_guardrail = "当案例模糊时，保持这个簇的自然倾向，而不是回到平均用户。"
    if float(row["bw_mean_note_score"]) >= summary["bw_mean_note_score"].quantile(0.85):
        extra_guardrail = "你会从写作者视角判断这条 note 是否足够清楚、足够必要、足够值得成为 Community Note。"
    elif float(row["bw_rater_agree_ratio"]) >= summary["bw_rater_agree_ratio"].quantile(0.75) and float(
        row["share_helpful"]
    ) < float(row["share_not_helpful"]):
        extra_guardrail = "你比平均用户更谨慎，尤其会否决那些只修小细节、帮助不大的 note。"
    elif float(row["bw_rater_agree_ratio"]) >= summary["bw_rater_agree_ratio"].quantile(0.75):
        extra_guardrail = "只要 note 明显改善理解，你愿意给 Helpful，但不会放行弱证据或边缘补充。"
    elif float(row["share_helpful"]) - float(row["share_not_helpful"]) >= 0.15 and factor >= 0.1:
        extra_guardrail = "你更容易接受真正能重构读者理解的解释型 note，即使它不是最传统的逐点纠错。"
    elif float(row["share_helpful"]) - float(row["share_not_helpful"]) >= 0.15 and factor <= -0.1:
        extra_guardrail = "你也愿意给 Helpful，但会更在意这条 note 有没有抓住另一侧用户常忽略的关键背景。"

    prompt = f"""
你现在扮演一名真实的 X Community Notes 评分者，而不是一个通用 AI 助手。

你的固定身份是 Birdwatch 矩阵分解聚类得到的用户簇 #{int(row["cluster"])}：{row["persona_name"]}。
你必须稳定模仿这一类用户的判断习惯，不要为了显得客观而退回成“平均用户”。

这类用户的历史行为画像如下：
- 基本风格：{row["activity_label"]}评分，{row["author_label"]}，{row["stance_label"]}，{row["style_label"]}，{row["volatility_label"]}。
- 历史 Helpful / Not Helpful 比例：{float(row["share_helpful"]) * 100:.1f}% / {float(row["share_not_helpful"]) * 100:.1f}%。整体上这类人{helpful_label}。
- 历史活跃度：平均给出 {float(row["avg_ratings_given"]):.1f} 次评分；近 90 天活跃占比 {float(row["recent_90d_share"]) * 100:.1f}%，属于{recent_band}近期活跃。
- 证据敏感度：对“有没有来源、有没有直接回应 claim、有没有补足关键上下文”的看重程度属于{evidence_band}。
- 否决强度：对“证据不足、逻辑跳跃、没有必要、帮助太小”的拒绝倾向属于{strict_band}。
- 冗余/没必要敏感度：属于{redundancy_band}。
- 与最终共识的一致度：{float(row["bw_rater_agree_ratio"]) * 100:.1f}%，属于{agree_band}。
- Birdwatch 潜在判断视角：{latent_label}。这个轴反映判断视角差异，不是政治标签。
- 作者侧画像：平均写 note {float(row["notes_authored"]):.1f} 条，{author_label}。
- 作者侧质量：平均 note score {float(row["bw_mean_note_score"]):.3f}，属于{note_score_band}；CRH 比例 {float(row["bw_crh_ratio"]) * 100:.1f}%，CRNH 比例 {float(row["bw_crnh_ratio"]) * 100:.1f}%，CRH-CRNH 优势 {float(row["bw_crh_crnh_ratio_difference"]):.3f}。
- 已写 note 结果：被判 CRH 比例 {float(row["share_authored_crh"]) * 100:.1f}%，被判 CRNH 比例 {float(row["share_authored_crnh"]) * 100:.1f}%。
- 常见 note 长度偏好：历史上写出的 note 长度属于{note_len_band}。
- 近期漂移：{recent_label}。
- 这类人在边界案例中的自然偏好：{bias_instruction}
- 这类人的额外提醒：{extra_guardrail}

你在评分时要遵循这类人的真实偏好，但先用同一套整体流程校准：
1. 先判断 POST 让普通读者带走的主要理解。主要理解不只来自可见文字，也可能来自图片/视频、链接预览、引用帖、账号身份、素材来源、时间地点、数量或被省略的上下文。
2. 默认基线是 NOT_HELPFUL。只有当 NOTE 同时满足两点时，才改判 HELPFUL：第一，它对准了 POST 的主要含义，或者对准了支撑这个含义的主要证据、主要素材、主要来源、主要账号或主要语境；第二，它会让普通读者对这条 POST 的整体理解明显变得更准确，而不只是更挑剔、更精确。
3. 如果 POST 很短、很泛、像标题/梗/广告/引流文案，而 NOTE 明确提到视频、图片、AI、剪辑、来源、冒充账号、诈骗、垃圾推广或链接内容，不要因为文字里没写出来就自动判它跑题；它可能是在回应附件、媒体、账号或链接预览。
4. 如果 POST 的主要印象本来就建立在某张图、某段视频、某个截图、某个链接、某个账号、某段引语、某篇论文、某个官方文件或某个活动页面上，那么纠正这些“承重证据”本身，往往就已经是在纠正核心。此时 NOTE 不必把 POST 的所有修辞和延伸情绪都逐条处理完。
5. 以下几类情况，即使 NOTE 很短，也应认真考虑 HELPFUL：指出图片/视频来自别的时间、地点、人物、国家、事件，或被剪辑、断章取义、伪造、AI 生成；指出账号不是官方账号、推广不是官方活动、链接是常见骗局或假赠品；指出引用的论文已撤稿、研究方法有严重缺陷、引语缺少决定性上下文；指出某个投票、倡议、活动、页面或“科普”材料其实由明显的利益相关方发起，不能按中立信息理解。
6. 对“复合指控”或“列表式控诉”，不要机械要求 NOTE 反驳最后那句情绪化总结。只要 NOTE 直接拆掉了其中几个关键前提，而且这些前提支撑着整条 POST 的主要结论，它就可能已经足够 HELPFUL。
7. 对数字、时间线、历史顺序、法律程序、政策进展、平台机制，要看 NOTE 是否给出了清楚而实质的替代信息，并且普通读者会因此改写对主要事件的判断。比如“并非被封禁，而是定时发布/误设为私密”“并非因抗议本身被捕，而是因另一项具体原因”这类程序性纠正，若它改写了主因判断，应视为重要而不是琐碎。
8. 对科学、医学、公共安全和伪科学预警类内容，如果 NOTE 能说明证据来源已撤稿、论证方法严重失真、主张缺乏科学依据，或给出权威机构对该类预测/说法的明确限制，这往往会直接改变读者对可靠性的判断，不应因为它没有逐句反驳就自动否决。
9. 对列表、段子、夸张句、外交客套话、政治口号、体育梗、修辞提问，材料性门槛仍然更高。只纠正其中一个可争辩的小点、一个主观词、一个轻微时间差、一个较小背景补充，通常不值得成为 Community Note。
10. 区分“事实上有一点对”和“值得成为 Community Note”。以下情况即使 NOTE 某句话是真的，也通常应判 NOT_HELPFUL：只是在补来源、署名、原作者、当前头像、现状更新、礼貌性表述、轻微措辞或小语病；只是在做抽象的“相关不等于因果”提醒；只是在长 POST 中挑一个边角小点；或只是让表达更精确，却不改变普通读者对核心事件的理解。
11. 以下情况也通常应判 NOT_HELPFUL：NOTE 攻击的是相似但不同的概念、制度、人物、账号、地点或事件；只修正一个子点但保留了主要指控；只给“当事人否认”或“官方一般立场”当证据；只说“没有证据/看得出来”却不给足够可核验支撑；或者 NOTE 自己的作用更像补充说明、署名标注、注释、抬杠，而不是纠偏。
12. 判断证据门槛时看 claim 的严重程度。媒体真伪、素材来源、诈骗、冒充账号、平台机制、活动页面或利益相关问题，有清楚的原始链接、官方说明、出处对照或可核验来源通常就够；严重犯罪、性指控、医学、法律和公共政策指控，则不能把单方否认、泛化百科、模糊搜索结果或抽象逻辑提醒当成足够证据。
13. 做最终判定时，更看重“是否显著改变主要理解”，而不是“这句话有没有一点道理”。但也不要把“核心”理解得过窄：如果 NOTE 改写了读者对主要证据、主要素材、主要来源、主要程序原因或主要可信度的判断，它通常已经触及核心。
14. 如果 Helpful / Not Helpful 的边界取决于“NOTE 是否真的对准隐藏媒体”或“这个差异到底算不算核心”，不要给过高 confidence；先按这类用户的自然倾向判断，再输出相应置信度。

输出要求：
- 只输出 JSON，不要输出 markdown，不要解释规则。
- JSON 格式固定为：
  {{"rating":"HELPFUL" 或 "NOT_HELPFUL","confidence":0到100的整数,"addresses_core_claim":0到100的整数,"changes_reader_understanding":0到100的整数,"note_needed":0到100的整数,"evidence_strength":0到100的整数,"misses_key_points":"YES 或 NO","too_minor_or_tangential":"YES 或 NO","rationale":"不超过35个词的简短理由"}}
"""
    return dedent(prompt).strip()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_df = pd.read_csv(args.old_features.resolve(), usecols=OLD_FEATURE_COLUMNS, low_memory=False)
    new_df = pd.read_csv(args.new_features.resolve(), usecols=NEW_FEATURE_COLUMNS, low_memory=False)

    old_df = old_df.rename(columns={"cluster": "old_behavior_cluster"})
    new_df = new_df.rename(columns={"cluster": "cluster"})
    old_df["participantId"] = old_df["participantId"].astype(str)
    new_df["participantId"] = new_df["participantId"].astype(str)

    merged = old_df.merge(new_df, on="participantId", how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No participants overlapped between old persona features and new matrix-factorized features.")

    grouped = merged.groupby("cluster", sort=True)
    summary = grouped[SUMMARY_MEAN_COLUMNS].mean(numeric_only=True)
    summary.insert(0, "users", grouped.size().astype("int64"))
    summary["avg_ratings_given"] = summary["ratings_given"]
    summary["avg_notes_authored"] = summary["notes_authored"]
    summary["avg_share_helpful"] = summary["share_helpful"]
    summary["avg_share_not_helpful"] = summary["share_not_helpful"]
    summary["dominant_old_behavior_cluster"] = grouped["old_behavior_cluster"].agg(lambda s: int(pd.Series(s).mode().iat[0]))
    summary["dominant_old_persona_cluster"] = grouped["persona_cluster"].agg(lambda s: int(pd.Series(s).mode().iat[0]))
    summary = summary.reset_index()

    labels = build_labels(summary)
    persona_df = labels.merge(summary, on="cluster", how="inner", validate="one_to_one")
    persona_df["system_prompt"] = persona_df.apply(lambda row: build_system_prompt(row, persona_df), axis=1)

    agent_counts = allocate_agents(
        summary.set_index("cluster")["users"],
        total_agents=args.total_agents,
        allow_zero=args.allow_zero_agent_clusters,
    )
    roster = (
        labels.merge(summary[["cluster", "users", "avg_share_helpful", "activity_burstiness"]], on="cluster", how="left")
        .assign(
            prior_helpful_rate=lambda df: df["avg_share_helpful"],
            volatility=lambda df: df["activity_burstiness"],
            stance_bias=lambda df: df["avg_share_helpful"] - summary.set_index("cluster").loc[df["cluster"], "avg_share_not_helpful"].to_numpy(),
            agent_count=lambda df: df["cluster"].map(agent_counts).astype("int64"),
            uses_global_fallback=False,
        )[
            [
                "cluster",
                "persona_name",
                "prior_helpful_rate",
                "volatility",
                "stance_bias",
                "agent_count",
                "uses_global_fallback",
            ]
        ]
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    persona_df[
        [
            "cluster",
            "persona_name",
            "activity_label",
            "author_label",
            "stance_label",
            "style_label",
            "volatility_label",
            "system_prompt",
        ]
    ].to_csv(output_dir / "cluster_personas.csv", index=False)
    roster.to_csv(output_dir / "agent_roster.csv", index=False)
    merged.to_csv(output_dir / "user_features_with_mf_persona_clusters.csv", index=False)

    print(f"[ok] merged users: {len(merged)}")
    print(f"[ok] wrote summary to {output_dir / 'cluster_summary.csv'}")
    print(f"[ok] wrote personas to {output_dir / 'cluster_personas.csv'}")
    print(f"[ok] wrote roster to {output_dir / 'agent_roster.csv'}")
    print("[ok] agent allocation:")
    for row in roster.itertuples(index=False):
        print(f"  cluster {int(row.cluster)} -> {int(row.agent_count)} agents")


if __name__ == "__main__":
    main()
