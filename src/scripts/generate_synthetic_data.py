"""
Deterministic generator for 100% SYNTHETIC sample datasets matching the real schemas.

Outputs:
- sample_data/synthetic_ki_offers.csv   (2,000 rows x 102 columns)
- sample_data/synthetic_ki_outcomes.csv (  640 rows x  68 columns)

All identifiers, names, clinical numbers, timestamps, and refusal fields are purely
synthetic and exist only so the ETL and evaluation pipelines can run end to end
without committing private medical records.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / 'sample_data'

OFFERS_COLUMNS = [
    'Unnamed: 0', 'Donor ID', 'Organ Txed', 'Donor ABO', 'KDPI', 'Donor Age (Yrs/Mths)',
    'PHS  Risk Factors', 'DCD', 'Match ID', 'Accept Sequence(s)', 'Match Classification',
    'Match Sequence', 'Hospital', 'Offer Accepted', 'Primary Refusal Code', 'Abo',
    'Allocation Dist', 'Anti Hbc', 'Anti Hcv', 'Birth Sex', 'BMI', 'Candidate Age At Offer',
    'clamp_dt_adj', 'Creatinine', 'Donor COD', 'Entire Name', 'EPTS pct', 'Final Response Dt',
    'Hbsag', 'Hbv Dna', 'Hcv Nat', 'Hgt Cm', 'hist IV druguse', 'init_creat', 'initial_response',
    'initial_response_dt', 'Lc Dist', 'Left Glumerul', 'Lt Ki Interstitial Fibrosis Descrip',
    'Lt Ki Pump', 'Lt Ki Vascular Changes Descrip', 'Match Submit Dt', 'notify_dt',
    'Other Specify Primary', 'Other Specify Secondary', 'peak_creat', 'Prime Descrip',
    'Recipient one cold isch', 'Recipient One Creat 1Y', 'Recipient One Creat 6M',
    'Recipient One Ctr', 'recipient one end EPTS', 'Recipient one first wk dial',
    'Recipient One Graft Status', 'Recipient one grf fail date', 'Recipient One Gtime',
    'Recipient one los', 'Recipient One Organ Txed', 'Recipient one organ',
    'Recipient One Patient Status', 'Recipient One Ptime', 'Recipient one pump ki',
    'Recipient one px stat date', 'Recipient one px stat', 'Recipient one rej',
    'Recipient One Serum Creat', 'Recipient One Tx Date', 'Recipient One Txorg',
    'Recipient two cold isch', 'Recipient Two Creat 1Y', 'Recipient Two Creat 6M',
    'Recipient Two Ctr', 'recipient two end EPTS', 'Recipient two first wk dial',
    'Recipient Two Graft Failure ', 'Recipient Two Graft Status', 'Recipient two grf fail date',
    'Recipient Two Gtime', 'Recipient two los', 'Recipient Two Organ Txed',
    'Recipient two organ', 'Recipient Two Patient Status', 'Recipient Two Ptime',
    'Recipient two pump ki', 'Recipient two px stat date', 'Recipient two px stat',
    'Recipient two rej', 'Recipient Two Serum Creat', 'Recipient Two Tx Date',
    'Recipient Two Txorg', 'recipient_one_PRE_creat', 'recipient_two_PRE_creat',
    'response_post_clamp', 'Right Glomerul', 'Rt Ki Interstitial Fibrosis Descrip',
    'Rt Ki Pump', 'Rt Ki Vascular Changes Descrip', 'Second Descrip',
    'Second Opo Refusal Id', 'Tx1 Dist', 'Tx2 Dist', 'Wgt Kg',
]

OUTCOMES_COLUMNS = [
    'Donor Id', 'Match Date', 'DCD', 'KDPI', 'Donor Age', 'PHS Risk Factors',
    'hist IV druguse', 'Birth Sex', 'Hospital with Offers', 'abo_donor', 'BMI',
    'Creatinine', 'Donor_COD', 'Hcv Nat', 'HEP B Core', 'HEP B NAT', 'HEP B Surface',
    'HEP C Serology', 'Hgt Cm', 'Init Creat', 'Left Glumerul',
    'Lt Ki Interstitial Fibrosis Descrip', 'Lt Ki Pump', 'Lt Ki Vascular Changes Descrip',
    'Peak Creat', 'Recipient one cold isch', 'Recipient One Creat 1Y',
    'Recipient One Creat 6M', 'Recipient One Ctr', 'recipient one end EPTS',
    'Recipient one first wk dial', 'Recipient One Graft Status',
    'Recipient one grf fail date', 'Recipient One Gtime', 'Recipient one los',
    'Recipient One Patient Status', 'Recipient One Pre Creat', 'Recipient One Ptime',
    'Recipient one pump ki', 'Recipient one px stat date', 'Recipient one rej',
    'Recipient One Serum Creat', 'Recipient One Txorg', 'Recipient two cold isch',
    'Recipient Two Creat 1Y', 'Recipient Two Creat 6M', 'Recipient Two Ctr',
    'recipient two end EPTS', 'Recipient two first wk dial',
    'Recipient Two Graft Status', 'Recipient two grf fail date', 'Recipient Two Gtime',
    'Recipient two los', 'Recipient Two Patient Status', 'Recipient Two Pre Creat',
    'Recipient Two Ptime', 'Recipient two pump ki', 'Recipient two px stat date',
    'Recipient two rej', 'Recipient Two Serum Creat', 'Recipient Two Txorg',
    'Right Glumerul', 'Rt Ki Interstitial Fibrosis Descrip', 'Rt Ki Pump',
    'Rt Ki Vascular Changes Descrip', 'Tx1 Dist', 'Tx2 Dist', 'Wgt Kg',
]

HOSPITALS = [f'SYNTH-H{i:02d}' for i in range(1, 16)]
HOSPITAL_NAMES = {
    h: f'SYNTHETIC Transplant Center {i:02d}' for i, h in enumerate(HOSPITALS, start=1)
}
HOSPITAL_WEIGHTS = np.array([
    0.28, 0.14, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01
])
HOSPITAL_WEIGHTS = HOSPITAL_WEIGHTS / HOSPITAL_WEIGHTS.sum()

BLOOD_TYPES = ['O', 'A', 'B', 'AB']
COD_VALUES = ['Anoxia', 'Head Trauma', 'Cerebrovascular/Stroke', 'CNS Tumor', 'Other Specify']
COND_DESCS = ['Absent', 'Minimal', 'Mild', 'Mild-moderate', 'None (<10%)', 'Unknown']
GLOM_PCTS = ['0-5%', '6-10%', '11-15%', '16-20%', 'Not Biopsied']
MATCH_CLASSES = [
    '250NM, 0%-20% EPTS, blood type identical or permissible',
    '250NM, blood type identical or permissible',
    '500NM, blood type identical or permissible',
    'National, blood type identical or permissible',
]
REFUSAL_CODES = [801, 802, 803, 810, 815, 820, 830, 835]


def build_synthetic_offers(rng: np.random.Generator) -> pd.DataFrame:
    n_donors = 40
    offers_per_donor = 50
    base_dt = datetime(2025, 1, 5, 8, 0, 0)
    rows: list[dict] = []
    row_idx = 0

    for d_idx in range(n_donors):
        donor_id = f'SYNTH_DONOR_{d_idx + 1:03d}'
        match_id = 900000 + d_idx + 1
        donor_abo = str(rng.choice(BLOOD_TYPES, p=[0.48, 0.34, 0.13, 0.05]))
        kdpi = int(rng.integers(5, 96))
        donor_age_yrs = int(rng.integers(18, 69))
        phs = str(rng.choice(['N', 'Y'], p=[0.8, 0.2]))
        dcd = str(rng.choice(['No', 'Yes'], p=[0.7, 0.3]))
        hgt = int(rng.integers(152, 196))
        wgt = float(np.round(rng.uniform(52.0, 115.0), 1))
        bmi = float(np.round(wgt / ((hgt / 100.0) ** 2), 2))
        init_creat = float(np.round(rng.uniform(0.6, 2.0), 2))
        peak_creat = float(np.round(init_creat + rng.uniform(0.1, 2.5), 2))
        creat = float(np.round(rng.uniform(init_creat, peak_creat), 2))
        birth_sex = str(rng.choice(['Male', 'Female']))
        cod = str(rng.choice(COD_VALUES))
        iv_drug = str(rng.choice(['N', 'Y', 'U'], p=[0.75, 0.2, 0.05]))
        anti_hbc = str(rng.choice(['N', 'P'], p=[0.92, 0.08]))
        anti_hcv = str(rng.choice(['N', 'P'], p=[0.9, 0.1]))
        hcv_nat = str(rng.choice(['N', 'P'], p=[0.94, 0.06]))
        lt_pump = str(rng.choice(['Y', 'N'], p=[0.55, 0.45]))
        rt_pump = str(rng.choice(['Y', 'N'], p=[0.55, 0.45]))
        lt_fib = str(rng.choice(COND_DESCS))
        rt_fib = str(rng.choice(COND_DESCS))
        lt_vasc = str(rng.choice(COND_DESCS))
        rt_vasc = str(rng.choice(COND_DESCS))
        lt_glom = str(rng.choice(GLOM_PCTS))
        rt_glom = str(rng.choice(GLOM_PCTS))

        match_submit_dt = base_dt + timedelta(days=int(d_idx * 4))
        clamp_dt = match_submit_dt + timedelta(hours=10)

        # Deterministic per-(donor, hospital) distance
        hosp_dists = {h: int(rng.integers(5, 650)) for h in HOSPITALS}

        # Pick 1 or 2 accepted sequences per donor (~2.8% acceptance rate)
        n_acc = 2 if d_idx % 3 == 0 else 1
        acc_positions = sorted(rng.choice(offers_per_donor, size=n_acc, replace=False).tolist())
        seqs = sorted(int(x) for x in rng.choice(range(1, 500), size=offers_per_donor, replace=False))
        acc_seq_l = seqs[acc_positions[0]]
        acc_seq_r = seqs[acc_positions[1]] if n_acc == 2 else None
        accept_str = f'{acc_seq_l} - L' + (f', {acc_seq_r} - R' if acc_seq_r is not None else '')

        # Synthetic recipient details for donor's accepted kidney(s)
        r1_ctr = HOSPITALS[int(rng.integers(0, len(HOSPITALS)))]
        r1_tx_dt = (clamp_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        r1_stat_dt = (clamp_dt + timedelta(days=180)).strftime('%Y-%m-%d')
        r1_cold = float(np.round(rng.uniform(8.0, 26.0), 2))
        r1_pre_creat = float(np.round(rng.uniform(4.0, 11.0), 2))
        r1_serum_creat = float(np.round(rng.uniform(0.9, 2.8), 2))
        r1_epts = float(int(rng.integers(5, 95)))
        r1_los = float(int(rng.integers(3, 14)))
        tx1_dist = hosp_dists[r1_ctr]

        if acc_seq_r is not None:
            r2_ctr = HOSPITALS[int(rng.integers(0, len(HOSPITALS)))]
            r2_tx_dt = (clamp_dt + timedelta(days=1)).strftime('%Y-%m-%d')
            r2_stat_dt = (clamp_dt + timedelta(days=180)).strftime('%Y-%m-%d')
            r2_cold = float(np.round(rng.uniform(8.0, 28.0), 2))
            r2_pre_creat = float(np.round(rng.uniform(4.0, 11.0), 2))
            r2_serum_creat = float(np.round(rng.uniform(0.9, 2.8), 2))
            r2_epts = float(int(rng.integers(5, 95)))
            r2_los = float(int(rng.integers(3, 14)))
            tx2_dist = float(hosp_dists[r2_ctr])
        else:
            r2_ctr = np.nan
            r2_tx_dt = np.nan
            r2_stat_dt = np.nan
            r2_cold = np.nan
            r2_pre_creat = np.nan
            r2_serum_creat = np.nan
            r2_epts = np.nan
            r2_los = np.nan
            tx2_dist = np.nan

        for pos, seq in enumerate(seqs):
            hosp = str(rng.choice(HOSPITALS, p=HOSPITAL_WEIGHTS))
            dist = hosp_dists[hosp]
            is_accepted = seq == acc_seq_l or seq == acc_seq_r
            init_resp = 'Z' if is_accepted or rng.random() < 0.85 else 'N'
            init_resp_dt = (match_submit_dt + timedelta(minutes=int(pos * 15 + 10))).strftime('%Y-%m-%d %H:%M:%S')
            final_resp_dt = (match_submit_dt + timedelta(minutes=int(pos * 15 + 45))).strftime('%Y-%m-%d %H:%M:%S')
            notify_dt = (
                (match_submit_dt + timedelta(minutes=int(pos * 15 + 5))).strftime('%Y-%m-%d %H:%M:%S')
                if rng.random() < 0.65
                else np.nan
            )

            if is_accepted:
                offer_acc = 'Yes'
                prim_code = np.nan
                prim_desc = np.nan
                sec_code = np.nan
                sec_desc = np.nan
                other_prim = np.nan
                other_sec = np.nan
            else:
                offer_acc = 'No'
                prim_code = float(rng.choice(REFUSAL_CODES))
                prim_desc = f'SYNTHETIC Refusal Code {int(prim_code)}'
                if rng.random() < 0.3:
                    sec_code = float(rng.choice(REFUSAL_CODES))
                    sec_desc = f'SYNTHETIC Secondary Code {int(sec_code)}'
                else:
                    sec_code = np.nan
                    sec_desc = np.nan
                other_prim = f'SYNTH_NOTE_{int(prim_code)}' if rng.random() < 0.25 else np.nan
                other_sec = f'SYNTH_SEC_NOTE_{int(prim_code)}' if rng.random() < 0.15 else np.nan

            row = {
                'Unnamed: 0': row_idx,
                'Donor ID': donor_id,
                'Organ Txed': 'Yes',
                'Donor ABO': donor_abo,
                'KDPI': kdpi,
                'Donor Age (Yrs/Mths)': f'{donor_age_yrs} (Yrs)',
                'PHS  Risk Factors': phs,
                'DCD': dcd,
                'Match ID': match_id,
                'Accept Sequence(s)': accept_str,
                'Match Classification': str(rng.choice(MATCH_CLASSES)),
                'Match Sequence': seq,
                'Hospital': hosp,
                'Offer Accepted': offer_acc,
                'Primary Refusal Code': prim_code,
                'Abo': str(rng.choice(BLOOD_TYPES, p=[0.48, 0.34, 0.13, 0.05])),
                'Allocation Dist': dist,
                'Anti Hbc': anti_hbc,
                'Anti Hcv': anti_hcv,
                'Birth Sex': birth_sex,
                'BMI': bmi,
                'Candidate Age At Offer': int(rng.integers(18, 76)),
                'clamp_dt_adj': clamp_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'Creatinine': creat,
                'Donor COD': cod,
                'Entire Name': HOSPITAL_NAMES[hosp],
                'EPTS pct': float(int(rng.integers(1, 100))),
                'Final Response Dt': final_resp_dt,
                'Hbsag': 'N',
                'Hbv Dna': 'N',
                'Hcv Nat': hcv_nat,
                'Hgt Cm': hgt,
                'hist IV druguse': iv_drug,
                'init_creat': init_creat,
                'initial_response': init_resp,
                'initial_response_dt': init_resp_dt,
                'Lc Dist': dist,
                'Left Glumerul': lt_glom,
                'Lt Ki Interstitial Fibrosis Descrip': lt_fib,
                'Lt Ki Pump': lt_pump,
                'Lt Ki Vascular Changes Descrip': lt_vasc,
                'Match Submit Dt': match_submit_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'notify_dt': notify_dt,
                'Other Specify Primary': other_prim,
                'Other Specify Secondary': other_sec,
                'peak_creat': peak_creat,
                'Prime Descrip': prim_desc,
                'Recipient one cold isch': r1_cold,
                'Recipient One Creat 1Y': np.nan,
                'Recipient One Creat 6M': float(np.round(r1_serum_creat * 0.95, 2)),
                'Recipient One Ctr': r1_ctr,
                'recipient one end EPTS': r1_epts,
                'Recipient one first wk dial': 'N',
                'Recipient One Graft Status': 'Functioning(180)',
                'Recipient one grf fail date': np.nan,
                'Recipient One Gtime': 180.0,
                'Recipient one los': r1_los,
                'Recipient One Organ Txed': 'KI',
                'Recipient one organ': 'KI',
                'Recipient One Patient Status': 'Alive(180)',
                'Recipient One Ptime': 180.0,
                'Recipient one pump ki': lt_pump,
                'Recipient one px stat date': r1_stat_dt,
                'Recipient one px stat': 'A',
                'Recipient one rej': np.nan,
                'Recipient One Serum Creat': r1_serum_creat,
                'Recipient One Tx Date': r1_tx_dt,
                'Recipient One Txorg': 'Left Kidney',
                'Recipient two cold isch': r2_cold,
                'Recipient Two Creat 1Y': np.nan,
                'Recipient Two Creat 6M': float(np.round(r2_serum_creat * 0.95, 2)) if acc_seq_r else np.nan,
                'Recipient Two Ctr': r2_ctr,
                'recipient two end EPTS': r2_epts,
                'Recipient two first wk dial': 'N' if acc_seq_r else np.nan,
                'Recipient Two Graft Failure ': 'No' if acc_seq_r else np.nan,
                'Recipient Two Graft Status': 'Functioning(180)' if acc_seq_r else np.nan,
                'Recipient two grf fail date': np.nan,
                'Recipient Two Gtime': 180.0 if acc_seq_r else np.nan,
                'Recipient two los': r2_los,
                'Recipient Two Organ Txed': 'KI' if acc_seq_r else np.nan,
                'Recipient two organ': 'KI' if acc_seq_r else np.nan,
                'Recipient Two Patient Status': 'Alive(180)' if acc_seq_r else np.nan,
                'Recipient Two Ptime': 180.0 if acc_seq_r else np.nan,
                'Recipient two pump ki': rt_pump if acc_seq_r else np.nan,
                'Recipient two px stat date': r2_stat_dt,
                'Recipient two px stat': 'A' if acc_seq_r else np.nan,
                'Recipient two rej': np.nan,
                'Recipient Two Serum Creat': r2_serum_creat,
                'Recipient Two Tx Date': r2_tx_dt,
                'Recipient Two Txorg': 'Right Kidney' if acc_seq_r else np.nan,
                'recipient_one_PRE_creat': r1_pre_creat,
                'recipient_two_PRE_creat': r2_pre_creat,
                'response_post_clamp': 'Post-Clamp' if pos % 3 == 0 else 'Pre-Clamp',
                'Right Glomerul': rt_glom,
                'Rt Ki Interstitial Fibrosis Descrip': rt_fib,
                'Rt Ki Pump': rt_pump,
                'Rt Ki Vascular Changes Descrip': rt_vasc,
                'Second Descrip': sec_desc,
                'Second Opo Refusal Id': sec_code,
                'Tx1 Dist': tx1_dist,
                'Tx2 Dist': tx2_dist,
                'Wgt Kg': wgt,
            }
            rows.append(row)
            row_idx += 1

    return pd.DataFrame(rows, columns=OFFERS_COLUMNS)


def build_synthetic_outcomes(rng: np.random.Generator) -> pd.DataFrame:
    n_donors = 80
    hospitals_per_donor = 8
    base_date = datetime(2025, 1, 1)
    rows: list[dict] = []

    for d_idx in range(n_donors):
        donor_id = f'SYNTH_OUT_DONOR_{d_idx + 1:03d}'
        match_dt = (base_date + timedelta(days=int(d_idx * 3))).strftime('%m/%d/%Y')
        hgt = int(rng.integers(152, 196))
        wgt = float(np.round(rng.uniform(52.0, 115.0), 1))
        bmi = float(np.round(wgt / ((hgt / 100.0) ** 2), 2))
        init_creat = float(np.round(rng.uniform(0.6, 2.0), 2))
        peak_creat = float(np.round(init_creat + rng.uniform(0.1, 2.5), 2))
        creat = float(np.round(rng.uniform(init_creat, peak_creat), 2))
        lt_pump = str(rng.choice(['Y', 'N']))
        rt_pump = str(rng.choice(['Y', 'N']))
        tx1_ctr = str(rng.choice(HOSPITALS))
        tx2_ctr = str(rng.choice(HOSPITALS))
        tx1_dist = int(rng.integers(5, 500))
        tx2_dist = int(rng.integers(5, 500))

        donor_template = {
            'Donor Id': donor_id,
            'Match Date': match_dt,
            'DCD': str(rng.choice(['No', 'Yes'], p=[0.7, 0.3])),
            'KDPI': str(int(rng.integers(5, 96))),
            'Donor Age': int(rng.integers(18, 69)),
            'PHS Risk Factors': str(rng.choice(['N', 'Y'], p=[0.8, 0.2])),
            'hist IV druguse': str(rng.choice(['N', 'Y', 'U'], p=[0.75, 0.2, 0.05])),
            'Birth Sex': str(rng.choice(['Male', 'Female'])),
            'abo_donor': str(rng.choice(BLOOD_TYPES, p=[0.48, 0.34, 0.13, 0.05])),
            'BMI': bmi,
            'Creatinine': creat,
            'Donor_COD': str(rng.choice(COD_VALUES)),
            'Hcv Nat': str(rng.choice(['N', 'P'], p=[0.94, 0.06])),
            'HEP B Core': str(rng.choice(['N', 'P'], p=[0.92, 0.08])),
            'HEP B NAT': 'N',
            'HEP B Surface': 'N',
            'HEP C Serology': str(rng.choice(['N', 'P'], p=[0.9, 0.1])),
            'Hgt Cm': hgt,
            'Init Creat': init_creat,
            'Left Glumerul': str(rng.choice(GLOM_PCTS)),
            'Lt Ki Interstitial Fibrosis Descrip': str(rng.choice(COND_DESCS)),
            'Lt Ki Pump': lt_pump,
            'Lt Ki Vascular Changes Descrip': str(rng.choice(COND_DESCS)),
            'Peak Creat': peak_creat,
            'Recipient one cold isch': float(np.round(rng.uniform(8.0, 26.0), 2)),
            'Recipient One Creat 1Y': float(np.round(rng.uniform(0.9, 2.5), 2)),
            'Recipient One Creat 6M': float(np.round(rng.uniform(0.9, 2.5), 2)),
            'Recipient One Ctr': tx1_ctr,
            'recipient one end EPTS': float(int(rng.integers(5, 95))),
            'Recipient one first wk dial': 'N',
            'Recipient One Graft Status': 'Functioning(365)',
            'Recipient one grf fail date': np.nan,
            'Recipient One Gtime': 365,
            'Recipient one los': float(int(rng.integers(3, 14))),
            'Recipient One Patient Status': 'Alive(365)',
            'Recipient One Pre Creat': float(np.round(rng.uniform(4.0, 11.0), 2)),
            'Recipient One Ptime': 365,
            'Recipient one pump ki': lt_pump,
            'Recipient one px stat date': '2026-01-01',
            'Recipient one rej': np.nan,
            'Recipient One Serum Creat': float(np.round(rng.uniform(0.9, 2.5), 2)),
            'Recipient One Txorg': 'Left Kidney',
            'Recipient two cold isch': float(np.round(rng.uniform(8.0, 26.0), 2)),
            'Recipient Two Creat 1Y': float(np.round(rng.uniform(0.9, 2.5), 2)),
            'Recipient Two Creat 6M': float(np.round(rng.uniform(0.9, 2.5), 2)),
            'Recipient Two Ctr': tx2_ctr,
            'recipient two end EPTS': float(int(rng.integers(5, 95))),
            'Recipient two first wk dial': 'N',
            'Recipient Two Graft Status': 'Functioning(365)',
            'Recipient two grf fail date': np.nan,
            'Recipient Two Gtime': 365,
            'Recipient two los': 6,
            'Recipient Two Patient Status': 'Alive(365)',
            'Recipient Two Pre Creat': float(np.round(rng.uniform(4.0, 11.0), 2)),
            'Recipient Two Ptime': 365,
            'Recipient two pump ki': rt_pump,
            'Recipient two px stat date': '2026-01-01',
            'Recipient two rej': np.nan,
            'Recipient Two Serum Creat': float(np.round(rng.uniform(0.9, 2.5), 2)),
            'Recipient Two Txorg': 'Right Kidney',
            'Right Glumerul': str(rng.choice(GLOM_PCTS)),
            'Rt Ki Interstitial Fibrosis Descrip': str(rng.choice(COND_DESCS)),
            'Rt Ki Pump': rt_pump,
            'Rt Ki Vascular Changes Descrip': str(rng.choice(COND_DESCS)),
            'Tx1 Dist': tx1_dist,
            'Tx2 Dist': tx2_dist,
            'Wgt Kg': wgt,
        }

        chosen_hospitals = rng.choice(HOSPITALS, size=hospitals_per_donor, replace=False)
        for hosp in chosen_hospitals:
            row = dict(donor_template)
            row['Hospital with Offers'] = str(hosp)
            rows.append(row)

    return pd.DataFrame(rows, columns=OUTCOMES_COLUMNS)


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    offers_df = build_synthetic_offers(rng)
    outcomes_df = build_synthetic_outcomes(rng)

    offers_path = SAMPLE_DIR / 'synthetic_ki_offers.csv'
    outcomes_path = SAMPLE_DIR / 'synthetic_ki_outcomes.csv'
    offers_df.to_csv(offers_path, index=False)
    outcomes_df.to_csv(outcomes_path, index=False)

    print(f'Wrote {offers_path.relative_to(PROJECT_ROOT)}: {offers_df.shape}')
    print(f'Wrote {outcomes_path.relative_to(PROJECT_ROOT)}: {outcomes_df.shape}')


if __name__ == '__main__':
    main()
