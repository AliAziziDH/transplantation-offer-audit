from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Column
from sqlmodel import Enum, Field, Relationship, SQLModel

from src.models.enums import Answer, BirthSex, BloodType, CauseOfDeath

if TYPE_CHECKING:
    from src.models.kidney import Kidney
    from src.models.offer import Offer


class Donor(SQLModel, table=True):
    age_years: int | None
    age_months: int | None
    kdpi: int
    bmi: float
    blood_type: BloodType = Field(sa_column=Column(Enum(BloodType)))
    creatinine: float
    init_creatinine: float | None
    peak_creatinine: float
    height_cm: int
    weight_kg: float
    clamp_dt: datetime

    phs_risk_factors: bool  # N or Y
    dcd: bool  # No or Yes
    hist_iv_drug_use: Answer = Field(sa_column=Column(Enum(Answer)))
    birth_sex: BirthSex = Field(sa_column=Column(Enum(BirthSex)))
    cod: CauseOfDeath = Field(sa_column=Column(Enum(CauseOfDeath)))

    # Test Results
    hbsag: bool  # N or P
    hbv_dna: bool  # N or P
    hcv_nat: bool  # N or P
    anti_hbc: bool  # N or P
    anti_hcv: bool  # N or P

    donor_id: str = Field(default=None, primary_key=True)

    # Relationships
    fk_left_kidney_id: int | None = Field(default=None, foreign_key='kidney.kidney_id')
    fk_right_kidney_id: int | None = Field(default=None, foreign_key='kidney.kidney_id')

    left_kidney: Optional["Kidney"] = Relationship(
        sa_relationship_kwargs={'primaryjoin': 'Donor.fk_left_kidney_id==Kidney.kidney_id', 'lazy': 'joined'})
    right_kidney: Optional["Kidney"] = Relationship(
        sa_relationship_kwargs={'primaryjoin': 'Donor.fk_right_kidney_id==Kidney.kidney_id', 'lazy': 'joined'})
    offers: Optional[List["Offer"]] = Relationship(
        back_populates='donor', sa_relationship_kwargs={'post_update': True})
