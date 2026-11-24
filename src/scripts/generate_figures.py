#!/usr/bin/env python3
"""
Generate publication-grade audit figures for the transplantation-offer-audit repository.

All plots render strictly from the aggregated offline benchmark statistics documented in
AUDIT.md and require zero access to private donor/candidate CSV or Excel files.
Figures use explicit light backgrounds (#FFFFFF / #F8FAFC) and dark slate typography so
they stay legible in both GitHub Light Mode and GitHub Dark Mode.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / 'assets'

BG_FIG = '#FFFFFF'
BG_AX = '#F8FAFC'
TEXT_DARK = '#0F172A'
TEXT_MUTED = '#334155'
GRID_COLOR = '#CBD5E1'
BORDER_COLOR = '#94A3B8'


def _configure_style() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.edgecolor': BORDER_COLOR,
        'axes.linewidth': 0.9,
        'figure.facecolor': BG_FIG,
        'axes.facecolor': BG_AX,
        'text.color': TEXT_DARK,
        'axes.labelcolor': TEXT_DARK,
        'xtick.color': TEXT_MUTED,
        'ytick.color': TEXT_MUTED,
    })


def _style_axis(ax: plt.Axes, grid_axis: str = 'x') -> None:
    ax.set_facecolor(BG_AX)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis=grid_axis, linestyle='--', alpha=0.55, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)


def make_leakage_waterfall() -> Path:
    """Create assets/leakage_waterfall.png comparing leaked vs clean accuracy."""
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(14.6, 6.8), gridspec_kw={'width_ratios': [1.35, 1.0]}
    )
    fig.patch.set_facecolor(BG_FIG)

    # --- Left Panel: ki_offers.csv -> Hospital (33 classes) ---
    labels_l = [
        'Claimed Headline\n(Unverified in Code)',
        'With Entire Name Leak\n(depth=10, diag)',
        'With Entire Name Leak\n(depth=5, ordinal)',
        'No Entire Name\n(99 cols, Random Split)',
        'Clean Clinical\n(30 cols, Random Split)',
        'Clean Clinical\n(30 cols, Donor GroupSplit)',
        'No Entire Name\n(99 cols, Donor GroupSplit)',
    ]
    test_acc = [99.30, 99.27, 88.72, 61.36, 32.71, 28.90, 21.11]
    bal_acc = [np.nan, 90.91, 45.45, 23.42, 7.55, 3.77, 11.79]

    y = np.arange(len(labels_l))
    bar_h = 0.36
    _style_axis(ax_l, grid_axis='x')

    colors_test = ['#94A3B8', '#DC2626', '#EA580C', '#D97706', '#2563EB', '#0D9488', '#0284C7']
    for i in range(len(labels_l)):
        if i == 0:
            ax_l.barh(
                y[i], test_acc[i], height=bar_h * 1.45, color='#E2E8F0',
                edgecolor='#475569', hatch='///', linewidth=1.1, zorder=3
            )
            ax_l.text(
                test_acc[i] + 1.2, y[i], f'{test_acc[i]:.2f}% (no Bal Acc)',
                va='center', ha='left', fontsize=8.4, fontweight='bold', color='#475569'
            )
        else:
            ax_l.barh(
                y[i] - bar_h / 2, test_acc[i], height=bar_h,
                color=colors_test[i], edgecolor='none', zorder=3
            )
            ax_l.barh(
                y[i] + bar_h / 2, bal_acc[i], height=bar_h,
                color='#64748B', alpha=0.72, edgecolor='none', zorder=3
            )
            ax_l.text(
                test_acc[i] + 1.2, y[i] - bar_h / 2, f'{test_acc[i]:.2f}%',
                va='center', ha='left', fontsize=8.3, fontweight='bold', color=TEXT_DARK,
                bbox=dict(facecolor=BG_AX, edgecolor='none', pad=0.4)
            )
            ax_l.text(
                bal_acc[i] + 1.2, y[i] + bar_h / 2, f'{bal_acc[i]:.2f}%',
                va='center', ha='left', fontsize=8.0, color='#475569',
                bbox=dict(facecolor=BG_AX, edgecolor='none', pad=0.4)
            )

    ax_l.axvline(29.83, color='#0F172A', linestyle='--', linewidth=1.3, zorder=2)
    ax_l.text(
        31.2, 6.72, 'Random Majority Baseline: 29.83%',
        va='center', ha='left', fontsize=8.1, fontweight='bold', color='#0F172A',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFFFFF', edgecolor='#CBD5E1', alpha=0.96),
        zorder=5
    )

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels_l, fontsize=8.8)
    ax_l.set_ylim(7.15, -0.65)
    ax_l.set_xlim(0, 136)
    ax_l.set_xlabel('Accuracy / Balanced Accuracy (%)', fontsize=9.8, fontweight='bold')
    ax_l.set_title(
        'A. ki_offers.csv -> Hospital (33 classes, N=59,970)\nAccuracy Collapses When Leaks Are Removed',
        fontsize=10.8, fontweight='bold', pad=10, color=TEXT_DARK
    )

    legend_handles = [
        mpatches.Patch(color='#2563EB', label='Test Accuracy (Clean / Leaked)'),
        mpatches.Patch(color='#64748B', alpha=0.72, label='Balanced Accuracy'),
        mpatches.Patch(facecolor='#E2E8F0', edgecolor='#475569', hatch='///', label='Unverified Claimed Headline'),
    ]
    ax_l.legend(
        handles=legend_handles, loc='lower right', fontsize=8.0,
        frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1'
    )

    # --- Right Panel: KI Outcome -> Hospital with Offers (35 classes) ---
    labels_r = [
        'Bayes Theoretical\nCeiling (1,003/8,348)',
        'Depth-5 GroupSplit\n(All 67 cols, Bal 4.19%)',
        'Test Majority Baseline\n(Random Split)',
        'Depth-5 Numeric\n(27 cols, Bal 3.36%)',
        'Depth-5 All Cols\n(Random, Bal 3.91%)',
    ]
    vals_r = [12.01, 8.01, 7.60, 7.07, 6.50]
    cols_r = ['#0D9488', '#0284C7', '#475569', '#3B82F6', '#60A5FA']

    yr = np.arange(len(labels_r))
    _style_axis(ax_r, grid_axis='x')

    ax_r.axvspan(12.01, 36.0, color='#FEE2E2', alpha=0.55, zorder=1)
    ax_r.axvline(12.01, color='#B91C1C', linestyle='--', linewidth=1.6, zorder=4)

    for i, (val, col) in enumerate(zip(vals_r, cols_r)):
        ax_r.barh(yr[i], val, height=0.50, color=col, zorder=3)
        ax_r.text(
            val + 0.55, yr[i], f'{val:.2f}%',
            va='center', ha='left', fontsize=8.6, fontweight='bold', color=TEXT_DARK,
            zorder=5
        )

    ax_r.text(
        24.2, 2.0,
        'Mathematically Impossible\nRegion (> 12.01%)\n\nClaimed 99.3% is 8.3x\nabove the Bayes limit\n(65/67 cols constant/donor)',
        ha='center', va='center', fontsize=8.7, fontweight='bold', color='#991B1B',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#FEF2F2', edgecolor='#FCA5A5', linewidth=1.1),
        zorder=5
    )

    ax_r.set_yticks(yr)
    ax_r.set_yticklabels(labels_r, fontsize=8.8)
    ax_r.invert_yaxis()
    ax_r.set_xlim(0, 36)
    ax_r.set_xlabel('Accuracy / Theoretical Ceiling (%)', fontsize=9.8, fontweight='bold')
    ax_r.set_title(
        'B. KI Outcome -> Hospital with Offers (35 classes, N=8,348)\nStructural Non-Identifiability Caps Accuracy at 12.01%',
        fontsize=10.8, fontweight='bold', pad=10, color=TEXT_DARK
    )

    fig.suptitle(
        'Deconstructing the "99.3% Accuracy" Claim (DecisionTree, test_size=0.4, random_state=42)',
        fontsize=13.2, fontweight='bold', y=0.99, color=TEXT_DARK
    )
    fig.text(
        0.5, 0.01,
        'Offline audit on private UNOS/OPTN extracts: ki_offers N=59,970 (Random 35,982/23,988; GroupSplit 39,222/20,748, 0 shared donors) | '
        'KI Outcome N=8,348 (Random 5,008/3,340; GroupSplit 5,102/3,246, 0 shared donors).',
        ha='center', va='bottom', fontsize=8.2, color='#475569'
    )

    plt.tight_layout(rect=[0.0, 0.045, 1.0, 0.94])
    out_path = ASSETS_DIR / 'leakage_waterfall.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight', facecolor=BG_FIG)
    plt.close(fig)
    return out_path


def make_base_rate_illusion() -> Path:
    """Create assets/base_rate_illusion.png illustrating Offer Accepted base-rate trap."""
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13.8, 5.9), gridspec_kw={'width_ratios': [0.95, 1.25]}
    )
    fig.patch.set_facecolor(BG_FIG)

    # --- Left Panel: Class Imbalance & Refusal Leak ---
    _style_axis(ax_l, grid_axis='x')
    classes = [
        'No (Declined)\n23,837 / 23,988 test\n(59,612 / 59,970 full)',
        'Yes (Accepted)\n151 / 23,988 test\n(358 / 59,970 full)',
    ]
    pcts = [99.37, 0.63]
    colors = ['#475569', '#0D9488']
    y_cls = np.arange(len(classes))

    ax_l.barh(y_cls, pcts, height=0.45, color=colors, zorder=3)
    ax_l.text(
        97.8, 0, '99.37% Test\n(99.40% Full)',
        va='center', ha='right', fontsize=9.1, fontweight='bold', color='#FFFFFF', zorder=4
    )
    ax_l.text(
        2.3, 1, '0.63% Test (0.60% Full)',
        va='center', ha='left', fontsize=9.1, fontweight='bold', color='#0F766E', zorder=4
    )

    ax_l.text(
        52.0, 1.56,
        'Post-Decision Leak in Full Feature Set:\n'
        'Primary Refusal Code & Prime Descrip are populated on\n'
        '59,612 / 59,612 No rows and NaN on 358 / 358 Yes rows\n'
        '-> Depth-1 tree hits 100.00% test accuracy in 2 leaves.',
        ha='center', va='center', fontsize=8.4, color='#991B1B', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#FEF2F2', edgecolor='#FCA5A5', linewidth=1.1),
        zorder=5
    )

    ax_l.set_yticks(y_cls)
    ax_l.set_yticklabels(classes, fontsize=8.8)
    ax_l.set_ylim(2.12, -0.55)
    ax_l.set_xlim(0, 108)
    ax_l.set_xlabel('Class Share in 40% Random Test Split (%)', fontsize=9.6, fontweight='bold')
    ax_l.set_title(
        'A. Severe Class Imbalance & Post-Decision Leak\n(Offer Accepted, N=59,970)',
        fontsize=10.6, fontweight='bold', pad=10, color=TEXT_DARK
    )

    # --- Right Panel: Raw Accuracy vs Minority Yes Detection ---
    _style_axis(ax_r, grid_axis='y')
    models = [
        "Always-'No'\nDummy",
        'Pre-Decision\n(35 cols, Ord,\nRandom)',
        'Pre-Decision\n(11 cols, Num,\nRandom)',
        'Pre-Decision\n(35 cols, Ord,\nGroupSplit)',
    ]
    raw_acc = [99.37, 99.35, 99.35, 94.90]
    yes_rec = [0.00, 17.22, 0.00, 14.08]
    yes_f1 = [0.00, 25.00, 0.00, 3.64]

    x = np.arange(len(models))
    w = 0.28

    b1 = ax_r.bar(x - w, raw_acc, width=w, color='#1D4ED8', label='Raw Test Accuracy (%)', zorder=3)
    b2 = ax_r.bar(x, yes_rec, width=w, color='#0D9488', label="'Yes' Recall (%)", zorder=3)
    b3 = ax_r.bar(x + w, yes_f1, width=w, color='#D97706', label="'Yes' F1 Score (%)", zorder=3)

    ax_r.axhline(
        99.37, color='#DC2626', linestyle='--', linewidth=1.3, zorder=2,
        label="Trivial Always-'No' Baseline (99.37%)"
    )

    for group_bars in (b1, b2, b3):
        for bar in group_bars:
            h = bar.get_height()
            is_inside = abs(h - 94.90) <= 0.5
            label_y = h - 7.5 if is_inside else h + 1.8
            txt_color = '#FFFFFF' if is_inside else TEXT_DARK
            fsize = 6.9 if is_inside else 7.3
            ax_r.text(
                bar.get_x() + bar.get_width() / 2, label_y,
                f'{h:.2f}%' if h >= 90 or h > 0 else '0.0%',
                ha='center', va='bottom', fontsize=fsize, fontweight='bold', color=txt_color,
                zorder=5
            )

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(models, fontsize=8.6)
    ax_r.set_ylim(0, 132)
    ax_r.set_ylabel('Score (%)', fontsize=9.6, fontweight='bold')
    ax_r.set_title(
        "B. Raw Accuracy vs. Minority 'Yes' Detection (max_depth=5)\n99.35% Pre-Decision Accuracy Is Below the Trivial Baseline",
        fontsize=10.6, fontweight='bold', pad=10, color=TEXT_DARK
    )
    ax_r.legend(
        loc='upper center', ncol=2, fontsize=7.9,
        frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1'
    )

    fig.suptitle(
        'The "99.35% Accuracy" Base-Rate Illusion on Binary Offer Acceptance',
        fontsize=13.0, fontweight='bold', y=0.99, color=TEXT_DARK
    )
    fig.text(
        0.5, 0.01,
        'Offline audit on ki_offers.csv (N=59,970): Random split N_train=35,982 / N_test=23,988 (151 Yes) | '
        'GroupShuffleSplit(Donor ID) N_train=39,222 (145 donors) / N_test=20,748 (97 donors, 142 Yes, 0 shared donors).',
        ha='center', va='bottom', fontsize=8.1, color='#475569'
    )

    plt.tight_layout(rect=[0.0, 0.045, 1.0, 0.94])
    out_path = ASSETS_DIR / 'base_rate_illusion.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight', facecolor=BG_FIG)
    plt.close(fig)
    return out_path


def make_leakage_channels_summary() -> Path:
    """Create assets/leakage_channels_summary.png showing single-col purity & donor overlap."""
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13.8, 5.9), gridspec_kw={'width_ratios': [1.28, 0.95]}
    )
    fig.patch.set_facecolor(BG_FIG)

    # --- Left Panel: Top Single-Column Bayes Purity for Hospital ---
    _style_axis(ax_l, grid_axis='x')
    cols = [
        'Entire Name (1-to-1 alias)',
        'Unnamed: 0 (row index)',
        'Final Response Dt (post-offer)',
        'initial_response_dt (batch ts)',
        'Allocation Dist / Lc Dist',
        'Other Specify Primary (refusal)',
        'Other Specify Secondary',
        'notify_dt (notification ts)',
        'Donor ID / Match ID',
    ]
    purities = [100.00, 100.00, 98.92, 86.34, 68.71, 49.29, 40.67, 36.52, 35.79]
    cat_colors = [
        '#DC2626', '#DC2626',
        '#D97706', '#D97706',
        '#2563EB',
        '#D97706', '#D97706', '#D97706',
        '#2563EB',
    ]

    yc = np.arange(len(cols))
    ax_l.barh(yc, purities, height=0.56, color=cat_colors, zorder=3)
    for i, val in enumerate(purities):
        ax_l.text(
            val + 1.1, yc[i], f'{val:.2f}%',
            va='center', ha='left', fontsize=8.3, fontweight='bold', color=TEXT_DARK, zorder=5
        )

    ax_l.axvline(29.66, color='#0F172A', linestyle='--', linewidth=1.3, zorder=2)
    ax_l.text(
        31.0, 9.0, 'Majority Baseline: 29.66%',
        va='center', ha='left', fontsize=8.0, fontweight='bold', color='#0F172A',
        bbox=dict(boxstyle='round,pad=0.24', facecolor='#FFFFFF', edgecolor='#CBD5E1', alpha=0.96),
        zorder=5
    )

    ax_l.set_yticks(yc)
    ax_l.set_yticklabels(cols, fontsize=8.6)
    ax_l.set_ylim(9.55, -0.6)
    ax_l.set_xlim(0, 118)
    ax_l.set_xlabel('Single-Column Bayes Purity for Hospital (%)', fontsize=9.6, fontweight='bold')
    ax_l.set_title(
        'A. Single-Column Target Purity (ki_offers.csv, N=59,970)\nWhy Uncleaned Features Memorize the Hospital Target',
        fontsize=10.6, fontweight='bold', pad=10, color=TEXT_DARK
    )

    legend_l = [
        mpatches.Patch(color='#DC2626', label='Direct Alias / Row Index (100%)'),
        mpatches.Patch(color='#D97706', label='Post-Offer Timestamps & Refusal Notes'),
        mpatches.Patch(color='#2563EB', label='Donor / Distance Proxy'),
    ]
    ax_l.legend(
        handles=legend_l, loc='lower right', fontsize=7.9,
        frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1'
    )

    # --- Right Panel: Donor Overlap Across Random vs Group Splits ---
    _style_axis(ax_r, grid_axis='y')
    datasets = ['ki_offers.csv\n(242 unique donors)', 'KI Outcome CSV\n(992 unique donors)']
    random_overlap = [90.08, 82.96]
    group_overlap = [0.00, 0.00]
    counts_labels = ['218 / 242\ndonors shared', '823 / 992\ndonors shared']

    xd = np.arange(len(datasets))
    w = 0.34
    br = ax_r.bar(xd - w / 2, random_overlap, width=w, color='#DC2626', label='Random Row Split (test_size=0.4)', zorder=3)
    bg = ax_r.bar(xd + w / 2, group_overlap, width=w, color='#0D9488', label='GroupShuffleSplit on Donor ID', zorder=3)

    for i, bar in enumerate(br):
        h = bar.get_height()
        ax_r.text(
            bar.get_x() + bar.get_width() / 2, h + 1.8,
            f'{h:.2f}%\n({counts_labels[i]})',
            ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#991B1B'
        )
    for bar in bg:
        ax_r.text(
            bar.get_x() + bar.get_width() / 2, 2.2,
            '0.00%\n(0 shared)',
            ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#0F766E'
        )

    ax_r.set_xticks(xd)
    ax_r.set_xticklabels(datasets, fontsize=9.2, fontweight='bold')
    ax_r.set_ylim(0, 126)
    ax_r.set_ylabel('Unique Donors Present in Both Train & Test (%)', fontsize=9.4, fontweight='bold')
    ax_r.set_title(
        'B. Donor-Level Train/Test Contamination\nRow-Level Random Splitting Leaks 83%-90% of Donors',
        fontsize=10.6, fontweight='bold', pad=10, color=TEXT_DARK
    )
    ax_r.legend(
        loc='upper right', fontsize=8.0, frameon=True,
        facecolor='#FFFFFF', edgecolor='#CBD5E1'
    )

    fig.suptitle(
        'Primary Leakage Channels Identified in the Repository Extracts',
        fontsize=13.0, fontweight='bold', y=0.99, color=TEXT_DARK
    )
    fig.text(
        0.5, 0.01,
        'Offline structural audit: ki_offers N=59,970 (avg 247.81 rows/donor) | '
        'KI Outcome N=8,348 (avg 8.42 rows/donor). GroupShuffleSplit enforces 0 donor overlap.',
        ha='center', va='bottom', fontsize=8.1, color='#475569'
    )

    plt.tight_layout(rect=[0.0, 0.045, 1.0, 0.94])
    out_path = ASSETS_DIR / 'leakage_channels_summary.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight', facecolor=BG_FIG)
    plt.close(fig)
    return out_path


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _configure_style()
    paths = [
        make_leakage_waterfall(),
        make_base_rate_illusion(),
        make_leakage_channels_summary(),
    ]
    for path in paths:
        print(f'Generated {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
