#!/usr/bin/env python3
"""
Leakage-aware evaluation and audit script for the Transplantation repository.

Reproduces the model hyperparameters from the original Transplantation_Model_01.py
(DecisionTreeClassifier(max_depth=5, random_state=42), test_size=0.4, random_state=42)
while fixing categorical encoding, evaluating accuracy / balanced accuracy / macro F1,
testing donor-grouped splits, and auditing target leakage and base-rate illusions.

Resolves all paths relative to PROJECT_ROOT (defaults to private files in data/ if
present locally, otherwise falls back to 100% synthetic sample files in sample_data/).
"""

from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_CLINICAL_COLS = [
    'Donor ABO',
    'KDPI',
    'Donor Age (Yrs/Mths)',
    'PHS  Risk Factors',
    'DCD',
    'Abo',
    'Anti Hbc',
    'Anti Hcv',
    'Birth Sex',
    'BMI',
    'Candidate Age At Offer',
    'Creatinine',
    'Donor COD',
    'EPTS pct',
    'Hbsag',
    'Hbv Dna',
    'Hcv Nat',
    'Hgt Cm',
    'hist IV druguse',
    'init_creat',
    'Left Glumerul',
    'Lt Ki Interstitial Fibrosis Descrip',
    'Lt Ki Pump',
    'Lt Ki Vascular Changes Descrip',
    'peak_creat',
    'Right Glomerul',
    'Rt Ki Interstitial Fibrosis Descrip',
    'Rt Ki Pump',
    'Rt Ki Vascular Changes Descrip',
    'Wgt Kg',
]


def resolve_input_path(arg_path: str | None, private_rel: str, sample_rel: str) -> Path:
    if arg_path:
        p = Path(arg_path)
        if p.is_absolute():
            return p
        if p.exists():
            return p.resolve()
        return (PROJECT_ROOT / p).resolve()
    private_path = PROJECT_ROOT / private_rel
    if private_path.is_file():
        return private_path
    return PROJECT_ROOT / sample_rel


def classify_source_label(path: Path, id_series: pd.Series) -> str:
    is_sample_dir = path.resolve().parent == (PROJECT_ROOT / 'sample_data').resolve()
    has_synth_ids = id_series.astype(str).str.startswith('SYNTH').any()
    if is_sample_dir or has_synth_ids:
        return '[SYNTHETIC SAMPLE DATA]'
    return '[LOCAL DATA]'


def print_header(title: str) -> None:
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


def single_column_bayes_purity(df: pd.DataFrame, target_col: str) -> list[tuple[str, float, int]]:
    n = len(df)
    results: list[tuple[str, float, int]] = []
    for col in df.columns:
        if col == target_col:
            continue
        max_correct = df.groupby(col, dropna=False)[target_col].apply(lambda s: s.value_counts().iloc[0]).sum()
        results.append((col, float(max_correct / n), int(df[col].nunique(dropna=False))))
    results.sort(key=lambda item: (-item[1], item[2], item[0]))
    return results


def audit_target_and_distributions(offers: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    print_header('1. TARGET & CLASS DISTRIBUTION AUDIT')

    has_hwo_offers = 'Hospital with Offers' in offers.columns
    has_hwo_outcomes = 'Hospital with Offers' in outcomes.columns
    print(f"Offers CSV shape: {offers.shape} | has 'Hospital with Offers'?: {has_hwo_offers}")
    print(f"Outcomes CSV shape: {outcomes.shape} | has 'Hospital with Offers'?: {has_hwo_outcomes}")

    h_vc = offers['Hospital'].value_counts(dropna=False)
    n_donors_offers = offers['Donor ID'].nunique()
    print(
        f"\n[Offers -> 'Hospital'] classes={len(h_vc)}, unique Donor IDs={n_donors_offers}, "
        f'avg rows/donor={len(offers)/n_donors_offers:.2f}'
    )
    print(f'  Majority class baseline: {h_vc.index[0]} = {h_vc.iloc[0]}/{len(offers)} ({h_vc.iloc[0]/len(offers):.4%})')
    print('  Top-5 classes:')
    for cls, cnt in h_vc.head(5).items():
        print(f'    {cls:12s}: {cnt:6d} ({cnt/len(offers):.4%})')
    print('  Bottom-3 classes:')
    for cls, cnt in h_vc.tail(3).items():
        print(f'    {cls:12s}: {cnt:6d} ({cnt/len(offers):.4%})')

    hwo_vc = outcomes['Hospital with Offers'].value_counts(dropna=False)
    n_donors_out = outcomes['Donor Id'].nunique()
    print(
        f"\n[Outcomes -> 'Hospital with Offers'] classes={len(hwo_vc)}, unique Donor Ids={n_donors_out}, "
        f'avg rows/donor={len(outcomes)/n_donors_out:.2f}'
    )
    print(
        f'  Majority class baseline: {hwo_vc.index[0]} = {hwo_vc.iloc[0]}/{len(outcomes)} '
        f'({hwo_vc.iloc[0]/len(outcomes):.4%})'
    )
    print('  Top-5 classes:')
    for cls, cnt in hwo_vc.head(5).items():
        print(f'    {cls:12s}: {cnt:6d} ({cnt/len(outcomes):.4%})')
    print('  Bottom-3 classes:')
    for cls, cnt in hwo_vc.tail(3).items():
        print(f'    {cls:12s}: {cnt:6d} ({cnt/len(outcomes):.4%})')

    oa_vc = offers['Offer Accepted'].value_counts(dropna=False)
    _, _, _, y_te_oa = train_test_split(
        offers.drop(columns=['Offer Accepted']),
        offers['Offer Accepted'],
        test_size=0.4,
        random_state=42,
    )
    oa_te_vc = y_te_oa.value_counts(dropna=False)
    print(f"\n[Offers -> 'Offer Accepted'] classes={len(oa_vc)}")
    for cls, cnt in oa_vc.items():
        print(f'  Full dataset {cls:3s}: {cnt:6d} ({cnt/len(offers):.4%})')
    for cls, cnt in oa_te_vc.items():
        print(f'  Test split (40%, rs=42) {cls:3s}: {cnt:6d} ({cnt/len(y_te_oa):.4%})')


def audit_leakage_and_separation(offers: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    print_header('2. DATA LEAKAGE & TRIVIAL SEPARATION AUDIT')

    pairs = offers[['Hospital', 'Entire Name']].drop_duplicates()
    print("[Offers -> 'Hospital']")
    print(
        f"  1-to-1 bijection check ('Hospital' <-> 'Entire Name'): "
        f"unique pairs={len(pairs)}, unique Hospital={offers['Hospital'].nunique()}, "
        f"unique Entire Name={offers['Entire Name'].nunique()}"
    )
    alloc_eq_lc = bool((offers['Allocation Dist'] == offers['Lc Dist']).all())
    donor_dist_unique = offers.groupby(['Donor ID', 'Allocation Dist'])['Hospital'].nunique()
    print(f'  Allocation Dist == Lc Dist on 100% of rows?: {alloc_eq_lc}')
    print(f'  Fraction of (Donor ID, Allocation Dist) pairs mapping to 1 Hospital: {(donor_dist_unique == 1).mean():.4%}')

    print("  Top-12 single-column Bayes purity for 'Hospital':")
    for col, purity, nuniq in single_column_bayes_purity(offers, 'Hospital')[:12]:
        print(f'    {col:38s} | purity={purity:.4%} | nunique={nuniq}')

    print("\n[Outcomes -> 'Hospital with Offers']")
    varying_cols = []
    for col in outcomes.columns:
        if col == 'Donor Id':
            continue
        nuniq_per_donor = outcomes.groupby('Donor Id')[col].nunique(dropna=False)
        n_varying = int((nuniq_per_donor > 1).sum())
        if n_varying > 0:
            varying_cols.append((col, n_varying, int(nuniq_per_donor.max())))
    print(f'  Columns varying within Donor Id (out of {len(outcomes.columns) - 1} non-ID cols): {varying_cols}')

    feat_cols = [c for c in outcomes.columns if c != 'Hospital with Offers']
    n_unique_x = outcomes.drop_duplicates(subset=feat_cols).shape[0]
    bayes_correct = int(
        outcomes.groupby(feat_cols, dropna=False)['Hospital with Offers']
        .apply(lambda s: s.value_counts().iloc[0])
        .sum()
    )
    print(f"  Unique feature vectors (excluding 'Hospital with Offers'): {n_unique_x} / {len(outcomes)}")
    print(f'  Exact Bayes-optimal theoretical ceiling: {bayes_correct}/{len(outcomes)} = {bayes_correct/len(outcomes):.4%}')

    print("\n[Offers -> 'Offer Accepted']")
    print('  Crosstab Offer Accepted vs Primary Refusal Code.isna():')
    print(pd.crosstab(offers['Offer Accepted'], offers['Primary Refusal Code'].isna()).to_string())
    print('  Crosstab Offer Accepted vs Prime Descrip.isna():')
    print(pd.crosstab(offers['Offer Accepted'], offers['Prime Descrip'].isna()).to_string())

    def seq_in_accept_list(row: pd.Series) -> bool:
        nums = [int(x) for x in re.findall(r'(\d+)\s*-\s*[LR]', str(row['Accept Sequence(s)']))]
        return int(row['Match Sequence']) in nums

    seq_match = offers.apply(seq_in_accept_list, axis=1)
    print('  Crosstab Offer Accepted vs (Match Sequence in Accept Sequence(s)):')
    print(pd.crosstab(offers['Offer Accepted'], seq_match).to_string())


def evaluate_tree(
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
    donor_ids: pd.Series,
    use_group_split: bool = False,
    encoding: str = 'ordinal',
    max_depth: int | None = 5,
    pos_label: str | None = None,
) -> None:
    if use_group_split:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=donor_ids))
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        d_tr, d_te = donor_ids.iloc[tr_idx], donor_ids.iloc[te_idx]
    else:
        X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
            X.copy(), y, donor_ids, test_size=0.4, random_state=42
        )

    shared_donors = len(set(d_tr) & set(d_te))

    cat_cols = X_tr.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    for c in num_cols:
        med = X_tr[c].median()
        fill_val = -999.0 if pd.isna(med) else med
        X_tr[c] = X_tr[c].fillna(fill_val)
        X_te[c] = X_te[c].fillna(fill_val)

    if encoding == 'ordinal':
        if cat_cols:
            enc = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(object))
            X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(object))
    elif encoding == 'onehot':
        if cat_cols:
            X_tr = pd.get_dummies(X_tr, columns=cat_cols, dummy_na=True)
            X_te = pd.get_dummies(X_te, columns=cat_cols, dummy_na=True)
            X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    elif encoding == 'numeric_only':
        X_tr = X_tr[num_cols]
        X_te = X_te[num_cols]
    else:
        raise ValueError(f'Unsupported encoding: {encoding}')

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_tr, y_tr)
    pred_tr = clf.predict(X_tr)
    pred_te = clf.predict(X_te)

    train_acc = accuracy_score(y_tr, pred_tr)
    test_acc = accuracy_score(y_te, pred_te)
    bal_acc = balanced_accuracy_score(y_te, pred_te)
    macro_f1 = f1_score(y_te, pred_te, average='macro', zero_division=0)
    test_maj = y_te.value_counts().iloc[0] / len(y_te)

    top_feats = pd.Series(clf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    top_feats = top_feats[top_feats > 0].head(3)
    top_str = ', '.join(f'{k}: {v:.4f}' for k, v in top_feats.items())

    print(
        f'[{label}] N_tr={len(X_tr)} ({d_tr.nunique()} donors) | N_te={len(X_te)} ({d_te.nunique()} donors, '
        f'shared_donors={shared_donors}) | depth={max_depth} | n_feat={X_tr.shape[1]} | '
        f'test_maj={test_maj:.4%} | train_acc={train_acc:.4%} | test_acc={test_acc:.4%} | '
        f'bal_acc={bal_acc:.4%} | macro_f1={macro_f1:.4f} | leaves={clf.get_n_leaves()}'
    )
    print(f'    top-3 features: {{{top_str}}}')

    if pos_label is not None:
        prec, rec, f1_pos, _ = precision_recall_fscore_support(
            y_te, pred_te, pos_label=pos_label, average='binary', zero_division=0
        )
        pred_pos = int((pred_te == pos_label).sum())
        actual_pos = int((y_te == pos_label).sum())
        print(
            f"    pos_class('{pos_label}'): precision={prec:.4%}, recall={rec:.4%}, "
            f'f1={f1_pos:.4f}, pred={pred_pos}/{actual_pos}'
        )


def run_evaluations(offers: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    print_header('3. DECISION TREE EVALUATIONS (test_size=0.4, random_state=42)')

    donor_offers = offers['Donor ID']
    donor_outcomes = outcomes['Donor Id']

    print('--- Group A: Offers CSV | Target = Hospital ---')
    y_h = offers['Hospital']
    X_h_with_leak = offers.drop(columns=['Hospital', 'Unnamed: 0'])
    evaluate_tree(X_h_with_leak, y_h, 'A1a: All cols minus Unnamed:0 (WITH Entire Name), ordinal', donor_offers)
    evaluate_tree(X_h_with_leak, y_h, 'A1b: All cols minus Unnamed:0 (WITH Entire Name), onehot', donor_offers, encoding='onehot')
    evaluate_tree(
        X_h_with_leak,
        y_h,
        'A1-depth-check (diag): WITH Entire Name, ordinal, max_depth=10',
        donor_offers,
        max_depth=10,
    )
    evaluate_tree(
        X_h_with_leak,
        y_h,
        'A1-depth-check (diag): WITH Entire Name, ordinal, max_depth=None',
        donor_offers,
        max_depth=None,
    )

    X_h_no_alias = offers.drop(columns=['Hospital', 'Entire Name', 'Unnamed: 0'])
    evaluate_tree(X_h_no_alias, y_h, 'A2a: Minus Entire Name & Unnamed:0, ordinal, random split', donor_offers)
    evaluate_tree(
        X_h_no_alias,
        y_h,
        'A2b: Minus Entire Name & Unnamed:0, numeric_only, random split',
        donor_offers,
        encoding='numeric_only',
    )
    evaluate_tree(
        X_h_no_alias,
        y_h,
        'A2c: Minus Entire Name & Unnamed:0, ordinal, GroupSplit(Donor ID)',
        donor_offers,
        use_group_split=True,
    )

    X_h_clean = offers[CLEAN_CLINICAL_COLS]
    evaluate_tree(X_h_clean, y_h, 'A3a: Clean pre-offer clinical features (30 cols), ordinal, random split', donor_offers)
    evaluate_tree(
        X_h_clean,
        y_h,
        'A3b: Clean pre-offer clinical features (30 cols), ordinal, GroupSplit(Donor ID)',
        donor_offers,
        use_group_split=True,
    )

    print('\n--- Group B: Outcomes CSV | Target = Hospital with Offers ---')
    y_out = outcomes['Hospital with Offers']
    X_out = outcomes.drop(columns=['Hospital with Offers'])
    evaluate_tree(X_out, y_out, 'B1a: All 67 cols, ordinal, random split', donor_outcomes)
    evaluate_tree(X_out, y_out, 'B1b: Numeric-only cols, random split', donor_outcomes, encoding='numeric_only')
    evaluate_tree(
        X_out,
        y_out,
        'B1c: All 67 cols, ordinal, GroupSplit(Donor Id)',
        donor_outcomes,
        use_group_split=True,
    )

    print('\n--- Group C: Offers CSV | Target = Offer Accepted (2 classes) ---')
    y_oa = offers['Offer Accepted']
    X_oa_all = offers.drop(columns=['Offer Accepted', 'Unnamed: 0'])
    evaluate_tree(
        X_oa_all,
        y_oa,
        'C1a: All cols minus Unnamed:0 (WITH refusal leak), ordinal',
        donor_offers,
        pos_label='Yes',
    )
    evaluate_tree(
        X_oa_all,
        y_oa,
        'C1b: Numeric-only median-imputed (WITH Primary Refusal Code), random split',
        donor_offers,
        encoding='numeric_only',
        pos_label='Yes',
    )

    pre_decision_cols = CLEAN_CLINICAL_COLS + [
        'Hospital',
        'Allocation Dist',
        'Match Classification',
        'Match Sequence',
        'response_post_clamp',
    ]
    X_oa_pre = offers[pre_decision_cols]
    evaluate_tree(
        X_oa_pre,
        y_oa,
        'C2a: Pre-decision features (35 cols), ordinal, random split',
        donor_offers,
        pos_label='Yes',
    )
    evaluate_tree(
        X_oa_pre,
        y_oa,
        'C2b: Pre-decision features, numeric_only, random split',
        donor_offers,
        encoding='numeric_only',
        pos_label='Yes',
    )
    evaluate_tree(
        X_oa_pre,
        y_oa,
        'C2c: Pre-decision features (35 cols), ordinal, GroupSplit(Donor ID)',
        donor_offers,
        use_group_split=True,
        pos_label='Yes',
    )


def main() -> None:
    parser = ArgumentParser(description='Audit target leakage and DecisionTree accuracy on kidney offer data.')
    parser.add_argument(
        '--offers-csv',
        default=None,
        help='Path to offers CSV (defaults to data/ki_offers.csv if present, else sample_data/synthetic_ki_offers.csv).',
    )
    parser.add_argument(
        '--outcomes-csv',
        default=None,
        help='Path to outcomes CSV (defaults to data/KI Outcome Download_data 10.1.20-9.30.23.csv if present, else sample_data/synthetic_ki_outcomes.csv).',
    )
    args = parser.parse_args()

    offers_path = resolve_input_path(args.offers_csv, 'data/ki_offers.csv', 'sample_data/synthetic_ki_offers.csv')
    outcomes_path = resolve_input_path(
        args.outcomes_csv,
        'data/KI Outcome Download_data 10.1.20-9.30.23.csv',
        'sample_data/synthetic_ki_outcomes.csv',
    )

    offers = pd.read_csv(offers_path, low_memory=False)
    outcomes = pd.read_csv(outcomes_path, low_memory=False)

    offers_rel = offers_path.relative_to(PROJECT_ROOT) if offers_path.is_relative_to(PROJECT_ROOT) else offers_path.name
    outcomes_rel = (
        outcomes_path.relative_to(PROJECT_ROOT) if outcomes_path.is_relative_to(PROJECT_ROOT) else outcomes_path.name
    )

    print_header('0. DATA SOURCE & TELEMETRY GROUNDING')
    print(f"Offers source  : {offers_rel} {classify_source_label(offers_path, offers['Donor ID'])} | N={len(offers)}")
    print(f"Outcomes source: {outcomes_rel} {classify_source_label(outcomes_path, outcomes['Donor Id'])} | N={len(outcomes)}")

    audit_target_and_distributions(offers, outcomes)
    audit_leakage_and_separation(offers, outcomes)
    run_evaluations(offers, outcomes)


if __name__ == '__main__':
    main()
