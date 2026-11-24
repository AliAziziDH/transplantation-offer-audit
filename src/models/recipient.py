from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Column, Enum
from sqlmodel import Field, Relationship, SQLModel

from src.models.enums import Answer, GraftStatus, PatientStatus

if TYPE_CHECKING:
    from src.models.kidney import Kidney
    from src.models.offer import Offer


class Recipient(SQLModel, table=True):

    cold_ischemic_time: float
    creatinine_1y: float | None
    creatinine_6m: float | None
    pre_creatinine: float
    serum_creatinine: float
    end_epts: float | None
    first_wk_dial: Answer = Field(sa_column=Column(Enum(Answer)))

    graft_status: GraftStatus = Field(sa_column=Column(Enum(GraftStatus)))
    graft_time: int | None
    graft_fail_date: date | None

    length_of_stay: int | None
    patient_status: PatientStatus = Field(sa_column=Column(Enum(PatientStatus)))
    patient_time: int | None
    patient_status_date: date | None
    rejection: Answer = Field(sa_column=Column(Enum(Answer)))
    transplant_date: date

    recipient_id: int | None = Field(default=None, primary_key=True)

    fk_offer_id: int | None = Field(default=None, foreign_key='offer.offer_id')

    offer: Optional["Offer"] = Relationship(back_populates='related_recipient')
    received_kidney: Optional["Kidney"] = Relationship(
        back_populates='recipient', sa_relationship_kwargs={'uselist': False, 'post_update': True})
