"""
Clean ETL loader for kidney offer CSV files into PostgreSQL (via SQLModel) or SQLite.

Supports:
- Relative path resolution via PROJECT_ROOT (no hardcoded user/OS paths).
- PostgreSQL mode via DBManager (when SQLModel + psycopg2 are installed and credentials are supplied).
- SQLite mode via ``--sqlite`` (defaults to ``:memory:``): uses SQLModel when available,
  or validates every row with the same ``read_*`` parser functions and loads built-in
  ``sqlite3`` normalized tables when ``sqlmodel`` is not installed.
"""

from argparse import ArgumentParser
from datetime import datetime
from os import environ
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Callable

from pandas import DataFrame, Series, isna, read_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.enums import (
    Answer,
    BirthSex,
    BloodType,
    CauseOfDeath,
    ConditionDescription,
    GraftStatus,
    PatientStatus,
    RefusalReasonType,
    Side,
)

try:
    from sqlmodel import Session, SQLModel  # type: ignore

    from src.database.config import DBManager
    from src.models.donor import Donor
    from src.models.hospital import Hospital
    from src.models.kidney import Kidney
    from src.models.offer import Offer
    from src.models.recipient import Recipient
    from src.models.refusal_reason import RefusalReason

    HAS_SQLMODEL = True
except ImportError:
    HAS_SQLMODEL = False


def resolve_default_csv(csv_arg: str | None) -> Path:
    if csv_arg:
        candidate = Path(csv_arg)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        return (PROJECT_ROOT / candidate).resolve()

    private_csv = PROJECT_ROOT / 'data' / 'ki_offers.csv'
    if private_csv.is_file():
        return private_csv
    return PROJECT_ROOT / 'sample_data' / 'synthetic_ki_offers.csv'


def read_age(age_input: str) -> tuple[int | None, int | None]:
    if isna(age_input) or not isinstance(age_input, str):
        return None, None
    m_result = re.match(r'(\d+).*\((Yrs|Mths)\)', age_input)
    if m_result:
        age_value = int(m_result.group(1))
        age_unit = m_result.group(2)
        if age_unit == 'Yrs':
            return age_value, None
        if age_unit == 'Mths':
            return None, age_value
    return None, None


def read_blood_type(s: str, /) -> BloodType:
    if isna(s) or not isinstance(s, str):
        return BloodType.UNKNOWN
    try:
        return getattr(BloodType, s.upper().strip())
    except AttributeError:
        return BloodType.UNKNOWN


def read_boolean(s: str | None, /, *, yes: str = 'Y', no: str = 'N') -> bool | None:
    if s is None or isna(s) or not isinstance(s, str) or not s.strip():
        return None
    val = s.upper().strip()
    if val == yes.upper():
        return True
    if val == no.upper():
        return False
    raise ValueError(f'Invalid boolean input: {s}. Expected {yes} or {no}.')


def read_discrete_answer(
    s: str | None,
    /,
    *,
    yes: str = 'Y',
    no: str = 'N',
    prov_yes: str = 'Z',
    uncertain: str = 'U',
) -> Answer:
    if s is None or isna(s) or not isinstance(s, str) or not s.strip():
        return Answer.UNCERTAIN
    val = s.upper().strip()
    if val == yes.upper().strip():
        return Answer.YES
    if val == no.upper().strip():
        return Answer.NO
    if val == prov_yes.upper().strip():
        return Answer.PROVISIONAL_YES
    if val == uncertain.upper().strip():
        return Answer.UNCERTAIN
    raise ValueError(
        f'Invalid input for discrete answer enum: {s}. '
        f'Expected {yes}, {no}, {prov_yes} or {uncertain}.'
    )


def read_birth_sex(s: str, /, *, male: str = 'MALE', female: str = 'FEMALE') -> BirthSex:
    val = str(s).upper().strip()
    if val == male.upper().strip():
        return BirthSex.MALE
    if val == female.upper().strip():
        return BirthSex.FEMALE
    raise ValueError(f'Invalid input for birth sex: {s}. Expected {male} or {female}.')


def read_cause_of_death(s: str, /) -> CauseOfDeath:
    val = str(s).upper().strip()
    if val == 'CEREBROVASCULAR/STROKE':
        return CauseOfDeath.CEREBROVASCULAR_STROKE
    if val == 'ANOXIA':
        return CauseOfDeath.ANOXIA
    if val == 'HEAD TRAUMA':
        return CauseOfDeath.HEAD_TRAUMA
    if val == 'CNS TUMOR':
        return CauseOfDeath.TUMOR
    if val == 'OTHER SPECIFY':
        return CauseOfDeath.OTHER
    raise ValueError(f'Invalid input for cause of death: {s}.')


def read_condition_description(s: str | None, /) -> ConditionDescription | None:
    if s is None or isna(s) or not isinstance(s, str) or not s.strip():
        return None
    val = s.upper().strip()
    if val == 'ABSENT':
        return ConditionDescription.ABSENT
    if val == 'MILD-MODERATE':
        return ConditionDescription.MILD_MODERATE
    if val == 'MILD':
        return ConditionDescription.MILD
    if val == 'MINIMAL':
        return ConditionDescription.MINIMAL
    if val == 'NONE (<10%)':
        return ConditionDescription.NONE
    if val == 'UNKNOWN':
        return ConditionDescription.UNKNOWN
    raise ValueError(f'Invalid input for condition description: {s}.')


def read_graft_status(
    s: str | None, /, *, functioning: str = 'FUNCTIONING', failed: str = 'FAILED'
) -> tuple[GraftStatus, int | None]:
    if s is None or isna(s) or not str(s).strip():
        return GraftStatus.UNKNOWN, None
    match = re.match(rf'({functioning}|{failed})\((\d+)\)', str(s).upper().strip())
    if match:
        status = match.group(1).upper().strip()
        days = int(match.group(2))
        if status == functioning:
            return GraftStatus.FUNCTIONING, days
        if status == failed:
            return GraftStatus.FAILED, days
    return GraftStatus.UNKNOWN, None


def read_patient_status(
    s: str | None, /, *, alive: str = 'ALIVE', deceased: str = 'DECEASED'
) -> tuple[PatientStatus, int | None]:
    if s is None or isna(s) or not str(s).strip():
        return PatientStatus.UNKNOWN, None
    match = re.match(rf'({alive}|{deceased})\((\d+)\)', str(s).upper().strip())
    if match:
        status = match.group(1).upper().strip()
        days = int(match.group(2))
        if status == alive:
            return PatientStatus.ALIVE, days
        if status == deceased:
            return PatientStatus.DECEASED, days
    return PatientStatus.UNKNOWN, None


def read_datetime(s: str | None, /, *, date_format: str = '%Y-%m-%d %H:%M:%S') -> datetime | None:
    if s is None or isna(s) or not isinstance(s, str) or not s.strip():
        return None
    return datetime.strptime(s.strip(), date_format)


def _clean_opt_str(val: Any) -> str | None:
    if val is None or isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def read_accept_sequences(row: Series) -> tuple[int | None, int | None]:
    raw = row['Accept Sequence(s)']
    s = '' if raw is None or isna(raw) else str(raw)
    left_match = re.search(r'(\d+)\s*-\s*L', s)
    right_match = re.search(r'(\d+)\s*-\s*R', s)
    left_value = int(left_match.group(1)) if left_match else None
    right_value = int(right_match.group(1)) if right_match else None
    return left_value, right_value


def parse_donor_fields(row: Series) -> dict[str, Any]:
    age_years, age_months = read_age(age_input=row['Donor Age (Yrs/Mths)'])
    return {
        'age_years': age_years,
        'age_months': age_months,
        'kdpi': int(row['KDPI']),
        'bmi': float(row['BMI']),
        'blood_type': read_blood_type(row['Donor ABO']),
        'creatinine': float(row['Creatinine']),
        'init_creatinine': None if isna(row['init_creat']) else float(row['init_creat']),
        'peak_creatinine': float(row['peak_creat']),
        'height_cm': int(row['Hgt Cm']),
        'weight_kg': float(row['Wgt Kg']),
        'clamp_dt': read_datetime(row['clamp_dt_adj']),
        'phs_risk_factors': read_boolean(row['PHS  Risk Factors']),
        'dcd': read_boolean(row['DCD'], yes='YES', no='NO'),
        'hist_iv_drug_use': read_discrete_answer(row['hist IV druguse']),
        'birth_sex': read_birth_sex(row['Birth Sex']),
        'cod': read_cause_of_death(row['Donor COD']),
        'hbsag': read_boolean(row['Hbsag'], yes='P', no='N'),
        'hbv_dna': read_boolean(row['Hbv Dna'], yes='P', no='N'),
        'hcv_nat': read_boolean(row['Hcv Nat'], yes='P', no='N'),
        'anti_hbc': read_boolean(row['Anti Hbc'], yes='P', no='N'),
        'anti_hcv': read_boolean(row['Anti Hcv'], yes='P', no='N'),
        'donor_id': str(row['Donor ID']).strip(),
    }


def parse_offer_fields(row: Series) -> dict[str, Any]:
    return {
        'candidate_blood_type': read_blood_type(row['Abo']),
        'candidate_age': int(row['Candidate Age At Offer']),
        'candidate_epts_pct': None if isna(row['EPTS pct']) else float(row['EPTS pct']),
        'initial_response': read_discrete_answer(row['initial_response']),
        'initial_response_dt': read_datetime(row['initial_response_dt']),
        'final_response_dt': read_datetime(row['Final Response Dt']),
        'notify_dt': read_datetime(row['notify_dt']),
        'match_submit_dt': read_datetime(row['Match Submit Dt']),
        'response_post_clamp': read_boolean(
            row['response_post_clamp'], yes='Post-Clamp', no='Pre-Clamp'
        ),
        'distance': int(row['Allocation Dist']),
        'specify_primary_refusal_reason': _clean_opt_str(row['Other Specify Primary']),
        'specify_secondary_refusal_reason': _clean_opt_str(row['Other Specify Secondary']),
        'match_id': int(row['Match ID']),
        'match_sequence': int(row['Match Sequence']),
    }


def parse_kidney_fields(row: Series, side: Side) -> dict[str, Any] | None:
    side_str = 'Lt' if side == Side.LEFT else 'Rt'
    pump = row[f'{side_str} Ki Pump']
    if_desc = read_condition_description(row[f'{side_str} Ki Interstitial Fibrosis Descrip'])
    vc_desc = read_condition_description(row[f'{side_str} Ki Vascular Changes Descrip'])
    has_pump = not isna(pump) and bool(str(pump).strip())
    if not has_pump and if_desc is None and vc_desc is None:
        return None
    return {
        'pump': read_discrete_answer(pump),
        'interstitial_fibrosis_description': if_desc,
        'vascular_changes_description': vc_desc,
    }


def parse_recipient_fields(row: Series, side: Side) -> dict[str, Any] | None:
    side_str0 = f'recipient {"one" if side == Side.LEFT else "two"}'
    side_str1 = f'Recipient {"one" if side == Side.LEFT else "two"}'
    side_str2 = f'Recipient {"One" if side == Side.LEFT else "Two"}'
    side_str3 = f'recipient_{"one" if side == Side.LEFT else "two"}'

    graft_status, graft_time = read_graft_status(row[f'{side_str2} Graft Status'])
    patient_status, patient_time = read_patient_status(row[f'{side_str2} Patient Status'])
    ci_time = row[f'{side_str1} cold isch']
    c_1y = row[f'{side_str2} Creat 1Y']
    c_6m = row[f'{side_str2} Creat 6M']
    epts = row[f'{side_str0} end EPTS']
    if (
        isna(ci_time)
        and isna(c_1y)
        and isna(c_6m)
        and isna(epts)
        and isna(graft_time)
        and isna(patient_time)
    ):
        return None

    los = row[f'{side_str1} los']
    gfail_dt = read_datetime(row[f'{side_str1} grf fail date'], date_format='%Y-%m-%d')
    px_stat_dt = read_datetime(row[f'{side_str1} px stat date'], date_format='%Y-%m-%d')
    tx_dt = read_datetime(row[f'{side_str2} Tx Date'], date_format='%Y-%m-%d')
    return {
        'cold_ischemic_time': float(ci_time),
        'creatinine_1y': None if isna(c_1y) else float(c_1y),
        'creatinine_6m': None if isna(c_6m) else float(c_6m),
        'pre_creatinine': float(row[f'{side_str3}_PRE_creat']),
        'serum_creatinine': float(row[f'{side_str2} Serum Creat']),
        'end_epts': None if isna(epts) else float(epts),
        'first_wk_dial': read_discrete_answer(row[f'{side_str1} first wk dial']),
        'graft_status': graft_status,
        'graft_time': graft_time,
        'graft_fail_date': gfail_dt.date() if gfail_dt else None,
        'length_of_stay': int(los) if not isna(los) else None,
        'patient_status': patient_status,
        'patient_time': patient_time,
        'patient_status_date': px_stat_dt.date() if px_stat_dt else None,
        'rejection': read_discrete_answer(row[f'{side_str1} rej']),
        'transplant_date': tx_dt.date() if tx_dt else None,
    }


def read_cacheable_item(
    row: Series,
    item_id_col: str,
    cache: dict,
    read_func: Callable[..., Any],
    apply_item_id: Callable[[Any], Any] | None = None,
    **kwargs,
):
    item_id = row[item_id_col]
    if item_id is None or isna(item_id) or (isinstance(item_id, str) and not item_id.strip()):
        return None
    if apply_item_id is not None:
        item_id = apply_item_id(item_id)
    if item_id in cache:
        return cache[item_id]
    item = read_func(row, **kwargs)
    cache[item_id] = item
    return item


def feed_offer_data_sqlmodel(df: DataFrame) -> dict[str, int]:
    refusal_reasons: dict[int, RefusalReason] = {}
    hospitals: dict[str, Hospital] = {}
    donors: dict[str, Donor] = {}
    kidneys: list[Kidney] = []
    recipients: list[Recipient] = []
    offers: list[Offer] = []

    def _mk_donor(row: Series) -> Donor:
        return Donor(**parse_donor_fields(row))

    def _mk_hospital(row: Series) -> Hospital:
        return Hospital(code=str(row['Hospital']).strip(), name=str(row['Entire Name']).strip())

    def _mk_refusal(row: Series, refusal_type: RefusalReasonType) -> RefusalReason | None:
        code_col = 'Primary Refusal Code' if refusal_type == RefusalReasonType.PRIMARY else 'Second Opo Refusal Id'
        desc_col = f'{"Prime" if refusal_type == RefusalReasonType.PRIMARY else "Second"} Descrip'
        code = row[code_col]
        if isna(code):
            return None
        return RefusalReason(code=int(code), description=str(row[desc_col]))

    engine = DBManager().get_engine()
    with Session(engine) as session:
        for _, row in df.iterrows():
            offer = Offer(**parse_offer_fields(row))
            offers.append(offer)
            donor = read_cacheable_item(row, 'Donor ID', donors, _mk_donor)
            hospital = read_cacheable_item(row, 'Hospital', hospitals, _mk_hospital)
            primary_ref = read_cacheable_item(
                row,
                'Primary Refusal Code',
                refusal_reasons,
                _mk_refusal,
                apply_item_id=int,
                refusal_type=RefusalReasonType.PRIMARY,
            )
            secondary_ref = read_cacheable_item(
                row,
                'Second Opo Refusal Id',
                refusal_reasons,
                _mk_refusal,
                apply_item_id=int,
                refusal_type=RefusalReasonType.SECONDARY,
            )

            left_acc, right_acc = read_accept_sequences(row)
            is_left = left_acc == offer.match_sequence
            is_right = right_acc == offer.match_sequence

            left_rec_d = parse_recipient_fields(row, Side.LEFT) if is_left else None
            left_kid_d = parse_kidney_fields(row, Side.LEFT) if is_left else None
            right_rec_d = parse_recipient_fields(row, Side.RIGHT) if is_right else None
            right_kid_d = parse_kidney_fields(row, Side.RIGHT) if is_right else None

            offer.donor = donor
            offer.hospital = hospital
            offer.primary_refusal_reason = primary_ref
            offer.secondary_refusal_reason = secondary_ref

            if left_kid_d is not None:
                left_kidney = Kidney(**left_kid_d)
                donor.left_kidney = left_kidney
                kidneys.append(left_kidney)
                if left_rec_d is not None:
                    left_rec = Recipient(**left_rec_d)
                    left_rec.received_kidney = left_kidney
                    left_rec.offer = offer
                    recipients.append(left_rec)
            if right_kid_d is not None:
                right_kidney = Kidney(**right_kid_d)
                donor.right_kidney = right_kidney
                kidneys.append(right_kidney)
                if right_rec_d is not None:
                    right_rec = Recipient(**right_rec_d)
                    right_rec.received_kidney = right_kidney
                    right_rec.offer = offer
                    recipients.append(right_rec)

        session.add_all(donors.values())
        session.add_all(hospitals.values())
        session.add_all(refusal_reasons.values())
        session.add_all(kidneys)
        session.add_all(recipients)
        session.add_all(offers)
        session.commit()

    return {
        'donor': len(donors),
        'hospital': len(hospitals),
        'refusal_reason': len(refusal_reasons),
        'kidney': len(kidneys),
        'offer': len(offers),
        'recipient': len(recipients),
    }


def feed_offer_data_sqlite3(df: DataFrame, db_target: str) -> dict[str, int]:
    """Fallback SQLite loader using stdlib sqlite3 and the exact same row parsers."""
    conn = sqlite3.connect(db_target)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS hospital (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refusal_reason (
            code INTEGER PRIMARY KEY,
            description TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS kidney (
            kidney_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pump TEXT NOT NULL,
            interstitial_fibrosis_description TEXT,
            vascular_changes_description TEXT,
            fk_recipient_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS donor (
            donor_id TEXT PRIMARY KEY,
            age_years INTEGER,
            age_months INTEGER,
            kdpi INTEGER NOT NULL,
            bmi REAL NOT NULL,
            blood_type TEXT NOT NULL,
            creatinine REAL NOT NULL,
            init_creatinine REAL,
            peak_creatinine REAL NOT NULL,
            height_cm INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            clamp_dt TEXT NOT NULL,
            phs_risk_factors INTEGER,
            dcd INTEGER,
            hist_iv_drug_use TEXT NOT NULL,
            birth_sex TEXT NOT NULL,
            cod TEXT NOT NULL,
            hbsag INTEGER,
            hbv_dna INTEGER,
            hcv_nat INTEGER,
            anti_hbc INTEGER,
            anti_hcv INTEGER,
            fk_left_kidney_id INTEGER,
            fk_right_kidney_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS offer (
            offer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_blood_type TEXT NOT NULL,
            candidate_age INTEGER NOT NULL,
            candidate_epts_pct REAL,
            initial_response TEXT NOT NULL,
            initial_response_dt TEXT NOT NULL,
            final_response_dt TEXT NOT NULL,
            notify_dt TEXT,
            match_submit_dt TEXT NOT NULL,
            response_post_clamp INTEGER,
            fk_donor_id TEXT,
            fk_hospital_code TEXT,
            distance INTEGER NOT NULL,
            fk_primary_refusal_reason_code INTEGER,
            fk_secondary_refusal_reason_code INTEGER,
            specify_primary_refusal_reason TEXT,
            specify_secondary_refusal_reason TEXT,
            match_id INTEGER NOT NULL,
            match_sequence INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recipient (
            recipient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cold_ischemic_time REAL NOT NULL,
            creatinine_1y REAL,
            creatinine_6m REAL,
            pre_creatinine REAL NOT NULL,
            serum_creatinine REAL NOT NULL,
            end_epts REAL,
            first_wk_dial TEXT NOT NULL,
            graft_status TEXT NOT NULL,
            graft_time INTEGER,
            graft_fail_date TEXT,
            length_of_stay INTEGER,
            patient_status TEXT NOT NULL,
            patient_time INTEGER,
            patient_status_date TEXT,
            rejection TEXT NOT NULL,
            transplant_date TEXT NOT NULL,
            fk_offer_id INTEGER
        );
        """
    )

    seen_donors: set[str] = set()
    seen_hospitals: set[str] = set()
    seen_refusals: set[int] = set()

    for _, row in df.iterrows():
        d = parse_donor_fields(row)
        donor_id = d['donor_id']
        if donor_id not in seen_donors:
            cur.execute(
                """
                INSERT INTO donor (
                    donor_id, age_years, age_months, kdpi, bmi, blood_type, creatinine,
                    init_creatinine, peak_creatinine, height_cm, weight_kg, clamp_dt,
                    phs_risk_factors, dcd, hist_iv_drug_use, birth_sex, cod,
                    hbsag, hbv_dna, hcv_nat, anti_hbc, anti_hcv
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    donor_id,
                    d['age_years'],
                    d['age_months'],
                    d['kdpi'],
                    d['bmi'],
                    d['blood_type'].name,
                    d['creatinine'],
                    d['init_creatinine'],
                    d['peak_creatinine'],
                    d['height_cm'],
                    d['weight_kg'],
                    d['clamp_dt'].isoformat() if d['clamp_dt'] else '',
                    None if d['phs_risk_factors'] is None else int(d['phs_risk_factors']),
                    None if d['dcd'] is None else int(d['dcd']),
                    d['hist_iv_drug_use'].name,
                    d['birth_sex'].name,
                    d['cod'].name,
                    None if d['hbsag'] is None else int(d['hbsag']),
                    None if d['hbv_dna'] is None else int(d['hbv_dna']),
                    None if d['hcv_nat'] is None else int(d['hcv_nat']),
                    None if d['anti_hbc'] is None else int(d['anti_hbc']),
                    None if d['anti_hcv'] is None else int(d['anti_hcv']),
                ),
            )
            seen_donors.add(donor_id)

        h_code = str(row['Hospital']).strip()
        if h_code not in seen_hospitals:
            cur.execute(
                'INSERT INTO hospital (code, name) VALUES (?, ?)',
                (h_code, str(row['Entire Name']).strip()),
            )
            seen_hospitals.add(h_code)

        prim_code = None if isna(row['Primary Refusal Code']) else int(row['Primary Refusal Code'])
        if prim_code is not None and prim_code not in seen_refusals:
            cur.execute(
                'INSERT INTO refusal_reason (code, description) VALUES (?, ?)',
                (prim_code, str(row['Prime Descrip'])),
            )
            seen_refusals.add(prim_code)

        sec_code = None if isna(row['Second Opo Refusal Id']) else int(row['Second Opo Refusal Id'])
        if sec_code is not None and sec_code not in seen_refusals:
            cur.execute(
                'INSERT INTO refusal_reason (code, description) VALUES (?, ?)',
                (sec_code, str(row['Second Descrip'])),
            )
            seen_refusals.add(sec_code)

        o = parse_offer_fields(row)
        cur.execute(
            """
            INSERT INTO offer (
                candidate_blood_type, candidate_age, candidate_epts_pct, initial_response,
                initial_response_dt, final_response_dt, notify_dt, match_submit_dt,
                response_post_clamp, fk_donor_id, fk_hospital_code, distance,
                fk_primary_refusal_reason_code, fk_secondary_refusal_reason_code,
                specify_primary_refusal_reason, specify_secondary_refusal_reason,
                match_id, match_sequence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                o['candidate_blood_type'].name,
                o['candidate_age'],
                o['candidate_epts_pct'],
                o['initial_response'].name,
                o['initial_response_dt'].isoformat() if o['initial_response_dt'] else '',
                o['final_response_dt'].isoformat() if o['final_response_dt'] else '',
                o['notify_dt'].isoformat() if o['notify_dt'] else None,
                o['match_submit_dt'].isoformat() if o['match_submit_dt'] else '',
                None if o['response_post_clamp'] is None else int(o['response_post_clamp']),
                donor_id,
                h_code,
                o['distance'],
                prim_code,
                sec_code,
                o['specify_primary_refusal_reason'],
                o['specify_secondary_refusal_reason'],
                o['match_id'],
                o['match_sequence'],
            ),
        )
        offer_id = cur.lastrowid

        left_acc, right_acc = read_accept_sequences(row)
        for is_acc, side, fk_col in (
            (left_acc == o['match_sequence'], Side.LEFT, 'fk_left_kidney_id'),
            (right_acc == o['match_sequence'], Side.RIGHT, 'fk_right_kidney_id'),
        ):
            if not is_acc:
                continue
            k = parse_kidney_fields(row, side)
            r = parse_recipient_fields(row, side)
            kid_id = None
            rec_id = None
            if k is not None:
                cur.execute(
                    """
                    INSERT INTO kidney (
                        pump, interstitial_fibrosis_description, vascular_changes_description
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        k['pump'].name,
                        k['interstitial_fibrosis_description'].name
                        if k['interstitial_fibrosis_description']
                        else None,
                        k['vascular_changes_description'].name
                        if k['vascular_changes_description']
                        else None,
                    ),
                )
                kid_id = cur.lastrowid
                cur.execute(f'UPDATE donor SET {fk_col} = ? WHERE donor_id = ?', (kid_id, donor_id))
            if r is not None:
                cur.execute(
                    """
                    INSERT INTO recipient (
                        cold_ischemic_time, creatinine_1y, creatinine_6m, pre_creatinine,
                        serum_creatinine, end_epts, first_wk_dial, graft_status, graft_time,
                        graft_fail_date, length_of_stay, patient_status, patient_time,
                        patient_status_date, rejection, transplant_date, fk_offer_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r['cold_ischemic_time'],
                        r['creatinine_1y'],
                        r['creatinine_6m'],
                        r['pre_creatinine'],
                        r['serum_creatinine'],
                        r['end_epts'],
                        r['first_wk_dial'].name,
                        r['graft_status'].name,
                        r['graft_time'],
                        r['graft_fail_date'].isoformat() if r['graft_fail_date'] else None,
                        r['length_of_stay'],
                        r['patient_status'].name,
                        r['patient_time'],
                        r['patient_status_date'].isoformat() if r['patient_status_date'] else None,
                        r['rejection'].name,
                        r['transplant_date'].isoformat() if r['transplant_date'] else '',
                        offer_id,
                    ),
                )
                rec_id = cur.lastrowid
                if kid_id is not None:
                    cur.execute(
                        'UPDATE kidney SET fk_recipient_id = ? WHERE kidney_id = ?',
                        (rec_id, kid_id),
                    )

    conn.commit()
    counts = {}
    for table in ('donor', 'hospital', 'refusal_reason', 'kidney', 'offer', 'recipient'):
        counts[table] = int(cur.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
    conn.close()
    return counts


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Load kidney offer CSV records into PostgreSQL or SQLite.')
    parser.add_argument(
        '--csv',
        default=None,
        help='Path to offers CSV (defaults to data/ki_offers.csv if present, else sample_data/synthetic_ki_offers.csv).',
    )
    parser.add_argument(
        '--sqlite',
        nargs='?',
        const=':memory:',
        default=None,
        help='Load into SQLite instead of PostgreSQL (optional file path; defaults to :memory:).',
    )
    parser.add_argument('--user', default=None, help='PostgreSQL username.')
    parser.add_argument('--password', default=None, help='PostgreSQL password.')
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    csv_path = resolve_default_csv(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f'Offers CSV not found: {csv_path}')

    df = read_csv(csv_path, low_memory=False)
    rel_csv = csv_path.relative_to(PROJECT_ROOT) if csv_path.is_relative_to(PROJECT_ROOT) else csv_path.name
    print(f'Loaded CSV: {rel_csv} ({len(df)} rows, {df.shape[1]} columns)')

    if args.sqlite is not None:
        if HAS_SQLMODEL:
            environ['DB_DBMS'] = 'sqlite'
            environ['DB_NAME'] = args.sqlite
            DBManager.reset()
            DBManager().create_db_and_tables()
            counts = feed_offer_data_sqlmodel(df)
            backend = f'sqlmodel+sqlite ({args.sqlite})'
        else:
            counts = feed_offer_data_sqlite3(df, args.sqlite)
            backend = f'sqlite3-fallback ({args.sqlite})'
    else:
        if not HAS_SQLMODEL:
            raise RuntimeError(
                'sqlmodel is required for PostgreSQL mode. Install requirements.txt or run with --sqlite :memory:.'
            )
        if args.user:
            environ['DB_USER'] = args.user
        if args.password:
            environ['DB_PASSWORD'] = args.password
        DBManager.reset()
        DBManager().create_db_and_tables()
        counts = feed_offer_data_sqlmodel(df)
        backend = 'sqlmodel+postgresql'

    print(f'Database load complete via {backend}: {counts}')


if __name__ == '__main__':
    main()
