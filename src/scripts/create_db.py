"""
Clean ETL loader for kidney offer CSV files into PostgreSQL or SQLite via SQLModel.

Supports:
- Relative path resolution via PROJECT_ROOT (no hardcoded user/OS paths).
- PostgreSQL mode via DBManager (when credentials are supplied or set in env).
- SQLite mode via ``--sqlite`` (defaults to ``:memory:``) using SQLModel + DBManager.
"""

from argparse import ArgumentParser
from datetime import datetime
from os import environ
from pathlib import Path
import re
import sys
from typing import Any, Callable

from pandas import DataFrame, Series, isna, read_csv
from sqlalchemy import func
from sqlmodel import Session, select

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.config import DBManager
from src.models.donor import Donor
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
from src.models.hospital import Hospital
from src.models.kidney import Kidney
from src.models.offer import Offer
from src.models.recipient import Recipient
from src.models.refusal_reason import RefusalReason


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
            name: int(session.exec(select(func.count()).select_from(model)).one())
            for name, model in (
                ('donor', Donor),
                ('hospital', Hospital),
                ('refusal_reason', RefusalReason),
                ('kidney', Kidney),
                ('offer', Offer),
                ('recipient', Recipient),
            )
        }


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
        environ['DB_DBMS'] = 'sqlite'
        environ['DB_NAME'] = args.sqlite
        DBManager.reset()
        DBManager().create_db_and_tables()
        counts = feed_offer_data_sqlmodel(df)
        backend = f'sqlmodel+sqlite ({args.sqlite})'
    else:
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
