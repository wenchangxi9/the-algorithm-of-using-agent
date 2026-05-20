from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_officialschema_nested_cv_aggregation_2000 import build_features

LABEL = {0: 'NOT_HELPFUL', 1: 'NEEDS_MORE_RATINGS', 2: 'HELPFUL'}
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=Path('artifacts/05_agent_runs_2000'))
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--acc-drop-max', type=float, default=0.004)
    p.add_argument('--min-nmr-recall', type=float, default=0.85)
    p.add_argument('--min-change-rate', type=float, default=0.002)
    return p.parse_args()


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    out = {'accuracy': float((y_true == y_pred).mean()), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred))}
    recs = []
    for label_id, label in LABEL.items():
        mask = y_true == label_id
        rec = float((y_pred[mask] == label_id).mean()) if mask.any() else math.nan
        out[f'recall_{label.lower()}'] = rec
        out[f'n_{label.lower()}'] = int(mask.sum())
        recs.append(rec)
    out['min_recall'] = float(np.nanmin(recs))
    out['h_to_nh'] = int(((y_true == 2) & (y_pred == 0)).sum())
    out['nh_to_h'] = int(((y_true == 0) & (y_pred == 2)).sum())
    out['cross_error'] = int(out['h_to_nh'] + out['nh_to_h'])
    return out


def score_key(m: dict[str, float | int], baseline: dict[str, float | int], objective: str = 'balanced') -> tuple[float, ...]:
    joint = 2.0 * m['accuracy'] * m['balanced_accuracy'] / max(m['accuracy'] + m['balanced_accuracy'], 1e-12)
    acc_delta = m['accuracy'] - baseline['accuracy']
    bal_delta = m['balanced_accuracy'] - baseline['balanced_accuracy']
    if objective == 'balanced':
        return (bal_delta, acc_delta, m['balanced_accuracy'], m['accuracy'], m['recall_not_helpful'], m['recall_helpful'], joint)
    if objective == 'accuracy':
        return (acc_delta, bal_delta, m['accuracy'], m['balanced_accuracy'], m['recall_not_helpful'], m['recall_helpful'], joint)
    raise ValueError(objective)


def add_meta_features(df: pd.DataFrame, fast_oof: pd.DataFrame) -> pd.DataFrame:
    fast_oof = fast_oof.copy()
    fast_oof['noteId'] = fast_oof['noteId'].astype(str)
    keep = ['noteId','nested_lr_summary_prob_not_helpful','nested_lr_summary_prob_nmr','nested_lr_summary_prob_helpful','nested_lr_full_agent_prob_not_helpful','nested_lr_full_agent_prob_nmr','nested_lr_full_agent_prob_helpful']
    fast_oof = fast_oof[[c for c in keep if c in fast_oof.columns]]
    df = df.copy(); df['noteId'] = df['noteId'].astype(str)
    df = df.merge(fast_oof, on='noteId', how='left')
    for prefix in ['nested_lr_summary', 'nested_lr_full_agent']:
        nh = df[f'{prefix}_prob_not_helpful'].astype(float)
        nmr = df[f'{prefix}_prob_nmr'].astype(float)
        h = df[f'{prefix}_prob_helpful'].astype(float)
        probs = np.vstack([nh.to_numpy(), nmr.to_numpy(), h.to_numpy()]).T
        safe = np.clip(probs, 1e-9, 1.0)
        df[f'{prefix}_entropy'] = -(safe * np.log(safe)).sum(axis=1)
        df[f'{prefix}_margin'] = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
        df[f'{prefix}_resolved_mass'] = nh + h
        df[f'{prefix}_signed_margin'] = h - nh
        df[f'{prefix}_nmr_gap'] = nmr - np.maximum(nh, h)
    return df


def build_views(feature_sets: dict[str, list[str]], meta_cols: list[str]) -> dict[str, list[str]]:
    return {'summary': feature_sets['summary'], 'summary_plus_meta': feature_sets['summary'] + meta_cols, 'full_plus_meta': feature_sets['full_agent_plus_metadata'] + meta_cols}


def make_model(c: float, class_weight, seed: int) -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', LogisticRegression(C=c, class_weight=class_weight, max_iter=5000, solver='lbfgs', random_state=seed))])


def align_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), p.shape[1]), dtype=float)
    for i, cls in enumerate(model.named_steps['clf'].classes_):
        out[:, int(cls)] = p[:, i]
    return out


def align_binary_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    if p.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    idx = int(np.where(model.named_steps['clf'].classes_ == 1)[0][0])
    return p[:, idx]


@dataclass(frozen=True)
class Spec:
    view: str
    c: float
    class_weight: object | None


def pick_binary_spec(X_train: pd.DataFrame, y_train: np.ndarray, views: dict[str, list[str]], specs: list[Spec], inner_folds: int, seed: int, objective: str) -> tuple[Spec, dict[str, float | int], np.ndarray]:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best = None; best_oof = None
    for spec in specs:
        cols = views[spec.view]
        oof_prob = np.zeros(len(y_train), dtype=float)
        for itr, iva in inner.split(X_train, y_train):
            model = make_model(spec.c, spec.class_weight, seed)
            model.fit(X_train.iloc[itr][cols], y_train[itr])
            oof_prob[iva] = align_binary_prob(model, X_train.iloc[iva][cols])
        pred = (oof_prob >= 0.5).astype(int)
        m = metric(y_train, pred)
        if best is None:
            best = (spec, m); best_oof = oof_prob; continue
        if score_key(m, best[1], objective) > score_key(best[1], m, objective):
            best = (spec, m); best_oof = oof_prob
    assert best is not None and best_oof is not None
    return best[0], best[1], best_oof


def fit_binary(X_tr, y_tr, X_te, cols, c, class_weight, seed):
    model = make_model(c, class_weight, seed)
    model.fit(X_tr[cols], y_tr)
    return align_binary_prob(model, X_te[cols])


def fit_mc(X_tr, y_tr, X_te, cols, c, class_weight, seed):
    model = make_model(c, class_weight, seed)
    model.fit(X_tr[cols], y_tr)
    return align_prob(model, X_te[cols])


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / 'officialschema_xstyle_group_gate_20260519'
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    pilot = pd.read_csv(run_dir / 'pilot_notes.csv', low_memory=False, dtype={'noteId': str})
    df = df.merge(pilot[['noteId', 'classification', 'topic_count']], on='noteId', how='left')
    if 'topic_count_x' in df.columns:
        df['topic_count'] = df['topic_count_x'].fillna(df.get('topic_count_y'))
        drop_cols = [c for c in ['topic_count_x', 'topic_count_y'] if c in df.columns]
        df = df.drop(columns=drop_cols)
    fast_oof = pd.read_csv(run_dir / 'officialschema_nested_cv_fast_20260512' / 'officialschema_nested_cv_fast_oof_predictions.csv')
    df = add_meta_features(df, fast_oof)
    meta_cols = [c for c in df.columns if c.startswith('nested_lr_')]
    views = build_views(feature_sets, meta_cols)
    y = df['true_label_3way'].to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    baseline_cols = views['full_plus_meta']
    base_pred = np.zeros(len(df), dtype=int)
    rescue_pred = np.zeros(len(df), dtype=int)
    rows = []

    specs = [Spec('summary_plus_meta', 0.3, 'balanced'), Spec('summary_plus_meta', 1.0, 'balanced'), Spec('full_plus_meta', 0.3, 'balanced'), Spec('full_plus_meta', 1.0, 'balanced'), Spec('full_plus_meta', 0.3, {0: 1.0, 1: 1.20})]

    def band_key(row):
        cls = str(row['classification']) if pd.notna(row['classification']) else 'UNKNOWN'
        tc = float(row['topic_count']) if pd.notna(row['topic_count']) else 0.0
        tc_band = 'high' if tc >= 3 else 'low' if tc <= 1 else 'mid'
        return f'{cls}__{tc_band}'

    df['gate_band'] = df.apply(band_key, axis=1)
    band_specs: dict[str, dict[str, float]] = {}

    for fold, (tr, te) in enumerate(outer.split(df, y), start=1):
        X_tr = df.iloc[tr].reset_index(drop=True)
        y_tr = y[tr]
        X_te = df.iloc[te].reset_index(drop=True)
        inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold)

        base_oof_prob = np.zeros((len(y_tr), 3), dtype=float)
        for itr, iva in inner.split(X_tr, y_tr):
            model = make_model(0.3, 'balanced', seed=args.seed + fold)
            model.fit(X_tr.iloc[itr][baseline_cols], y_tr[itr])
            base_oof_prob[iva] = align_prob(model, X_tr.iloc[iva][baseline_cols])
        base_oof_pred = base_oof_prob.argmax(axis=1)
        base_inner_m = metric(y_tr, base_oof_pred)

        resolved_y = (y_tr != 1).astype(int)
        resolved_spec, resolved_inner_m, resolved_oof = pick_binary_spec(X_tr, resolved_y, views, specs, args.inner_folds, args.seed + 11 + fold, 'balanced')
        resolved_mask = y_tr != 1
        direction_y = (y_tr[resolved_mask] == 2).astype(int)
        direction_spec, direction_inner_m, direction_oof_resolved = pick_binary_spec(X_tr.iloc[resolved_mask].reset_index(drop=True), direction_y, views, specs, max(2, min(args.inner_folds, 4)), args.seed + 23 + fold, 'balanced')
        direction_oof = np.full(len(y_tr), 0.5, dtype=float); direction_oof[resolved_mask] = direction_oof_resolved

        # Learn group-specific gates in training only.
        group_rows = []
        for band in sorted(X_tr['gate_band'].astype(str).unique()):
            band_mask = X_tr['gate_band'].astype(str).to_numpy() == band
            if band_mask.sum() < 30:
                continue
            y_band = y_tr[band_mask]
            bp = base_oof_prob[band_mask]
            rp = resolved_oof[band_mask]
            dp = direction_oof[band_mask]
            baseline_nmr = bp[:, 1]
            baseline_pred_band = base_oof_pred[band_mask]
            baseline_acc_floor = metric(y_band, baseline_pred_band)['accuracy'] - args.acc_drop_max
            best = None
            for t_resolved in np.linspace(0.30, 0.80, 11):
                for t_margin in np.linspace(-0.10, 0.24, 9):
                    for t_conf in np.linspace(0.50, 0.80, 7):
                        pred = baseline_pred_band.copy()
                        rescue = (baseline_pred_band == 1) & (rp >= t_resolved) & ((rp - baseline_nmr) >= t_margin) & (np.maximum(dp, 1.0 - dp) >= t_conf)
                        if rescue.mean() < args.min_change_rate:
                            continue
                        pred[rescue] = np.where(dp[rescue] >= 0.5, 2, 0)
                        m = metric(y_band, pred)
                        if m['accuracy'] < baseline_acc_floor or m['recall_needs_more_ratings'] < args.min_nmr_recall:
                            continue
                        key = (m['balanced_accuracy'], m['accuracy'], m['recall_not_helpful'], m['recall_helpful'])
                        if best is None or key > best[0]:
                            best = (key, {'t_resolved': float(t_resolved), 't_margin': float(t_margin), 't_conf': float(t_conf)}, m)
            if best is not None:
                band_specs[band] = best[1]
                group_rows.append({'fold': fold, 'band': band, **best[1], **{f'inner_{k}': v for k, v in best[2].items()}})
            else:
                band_specs[band] = {'t_resolved': 1.1, 't_margin': 1.1, 't_conf': 1.1}
                group_rows.append({'fold': fold, 'band': band, 't_resolved': 1.1, 't_margin': 1.1, 't_conf': 1.1})

        base_model = make_model(0.3, 'balanced', seed=args.seed + fold)
        base_model.fit(X_tr[baseline_cols], y_tr)
        base_prob_te = align_prob(base_model, X_te[baseline_cols])
        base_pred_te = base_prob_te.argmax(axis=1)

        resolved_model = make_model(resolved_spec.c, resolved_spec.class_weight, seed=args.seed + 101 + fold)
        resolved_model.fit(X_tr[views[resolved_spec.view]], (y_tr != 1).astype(int))
        resolved_prob_te = align_binary_prob(resolved_model, X_te[views[resolved_spec.view]])

        direction_model = make_model(direction_spec.c, direction_spec.class_weight, seed=args.seed + 202 + fold)
        direction_model.fit(X_tr[views[direction_spec.view]][resolved_mask], (y_tr[resolved_mask] == 2).astype(int))
        direction_prob_te = align_binary_prob(direction_model, X_te[views[direction_spec.view]])

        final_te = base_pred_te.copy()
        te_band = X_te['gate_band'].astype(str).to_numpy()
        for band in np.unique(te_band):
            band_mask = te_band == band
            spec = band_specs.get(band, {'t_resolved': 1.1, 't_margin': 1.1, 't_conf': 1.1})
            rescue = (base_pred_te[band_mask] == 1) & (resolved_prob_te[band_mask] >= spec['t_resolved']) & ((resolved_prob_te[band_mask] - base_prob_te[band_mask, 1]) >= spec['t_margin']) & (np.maximum(direction_prob_te[band_mask], 1.0 - direction_prob_te[band_mask]) >= spec['t_conf'])
            final_te[np.where(band_mask)[0][rescue]] = np.where(direction_prob_te[band_mask][rescue] >= 0.5, 2, 0)

        base_pred[te] = base_pred_te
        rescue_pred[te] = final_te
        rows.append({'fold': fold, 'resolved_view': resolved_spec.view, 'resolved_c': resolved_spec.c, 'resolved_class_weight': resolved_spec.class_weight if isinstance(resolved_spec.class_weight, str) else json.dumps(resolved_spec.class_weight), 'direction_view': direction_spec.view, 'direction_c': direction_spec.c, 'direction_class_weight': direction_spec.class_weight if isinstance(direction_spec.class_weight, str) else json.dumps(direction_spec.class_weight), **{f'inner_baseline_{k}': v for k, v in base_inner_m.items()}, **{f'inner_resolved_{k}': v for k, v in resolved_inner_m.items()}, **{f'inner_direction_{k}': v for k, v in direction_inner_m.items()}, **{f'test_baseline_{k}': v for k, v in metric(y[te], base_pred_te).items()}, **{f'test_rescue_{k}': v for k, v in metric(y[te], final_te).items()}})

    preds = df[['noteId', 'true_label_3way', 'true_label_text']].copy(); preds['baseline_full_plus_meta_lr_balanced'] = base_pred; preds['xstyle_group_gate'] = rescue_pred
    summary = pd.DataFrame([
        {'method': 'baseline_full_plus_meta_lr_balanced', 'family': 'baseline', 'feature_set': 'full_plus_meta', **metric(y, base_pred)},
        {'method': 'xstyle_group_gate', 'family': 'xstyle_gate', 'feature_set': 'baseline_plus_group_thresholds', **metric(y, rescue_pred)},
    ])
    for c in ['accuracy', 'balanced_accuracy', 'recall_not_helpful', 'recall_needs_more_ratings', 'recall_helpful', 'min_recall']:
        summary[f'{c}_pct'] = pd.to_numeric(summary[c], errors='coerce') * 100.0
    summary = summary.sort_values(['accuracy', 'balanced_accuracy', 'min_recall'], ascending=[False, False, False])
    best = str(summary.iloc[0]['method'])
    confusion = pd.crosstab(preds['true_label_text'], preds[best].map(LABEL), margins=True)
    preds.to_csv(out_dir / 'oof_predictions.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(out_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(out_dir / 'fold_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame.from_dict(band_specs, orient='index').reset_index(names='band').to_csv(out_dir / 'band_specs.csv', index=False, encoding='utf-8-sig')
    confusion.to_csv(out_dir / 'best_confusion.csv', encoding='utf-8-sig')
    (out_dir / 'run_metadata.json').write_text(json.dumps({'run_dir': str(run_dir), 'out_dir': str(out_dir), 'n_notes': int(len(df)), 'folds': int(args.folds), 'inner_folds': int(args.inner_folds), 'seed': int(args.seed), 'acc_drop_max': float(args.acc_drop_max), 'min_nmr_recall': float(args.min_nmr_recall), 'min_change_rate': float(args.min_change_rate), 'best': summary.iloc[0].to_dict(), 'feature_sets': {k: len(v) for k, v in views.items()}, 'band_specs': band_specs, 'resolved_spec': {'view': resolved_spec.view, 'c': resolved_spec.c, 'class_weight': resolved_spec.class_weight if isinstance(resolved_spec.class_weight, str) else json.dumps(resolved_spec.class_weight)}, 'direction_spec': {'view': direction_spec.view, 'c': direction_spec.c, 'class_weight': direction_spec.class_weight if isinstance(direction_spec.class_weight, str) else json.dumps(direction_spec.class_weight)}}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[['method', 'accuracy_pct', 'balanced_accuracy_pct', 'recall_not_helpful_pct', 'recall_needs_more_ratings_pct', 'recall_helpful_pct']].to_string(index=False))
    print('\nBest confusion matrix:')
    print(confusion.to_string())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
