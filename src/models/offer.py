from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Column, Enum
from sqlmodel import Field, Relationship, SQLModel

from src.models.enums import Answer, BloodType

if TYPE_CHECKING:
    from src.models.donor import Donor
    from src.models.hospital import Hospital
    from src.models.recipient import Recipient
    from src.models.refusal_reason import RefusalReason


class Offer(SQLModel, table=True):

    candidate_blood_type: BloodType = Field(sa_column=Column(Enum(BloodType)))
    candidate_age: int
    candidate_epts_pct: float | None
    initial_response: Answer = Field(sa_column=Column(Enum(Answer)))
    initial_response_dt: datetime
    final_response_dt: datetime
    notify_dt: datetime | None
    match_submit_dt: datetime
    response_post_clamp: bool

    fk_donor_id: str | None = Field(default=None, foreign_key='donor.donor_id')
    fk_hospital_code: str | None = Field(default=None, foreign_key='hospital.code')
    distance: int
    fk_primary_refusal_reason_code: int | None = Field(
        default=None, foreign_key='refusal_reason.code')
    fk_secondary_refusal_reason_code: int | None = Field(
        default=None, foreign_key='refusal_reason.code')
    specify_primary_refusal_reason: str | None
    specify_secondary_refusal_reason: str | None

    match_id: int
    match_sequence: int
    offer_id: int | None = Field(default=None, primary_key=True)

    # Relationships
    donor: Optional["Donor"] = Relationship(back_populates='offers')
    related_recipient: Optional["Recipient"] = Relationship(back_populates='offer')
    hospital: Optional["Hospital"] = Relationship()
    primary_refusal_reason: Optional["RefusalReason"] = Relationship(
        sa_relationship_kwargs={'primaryjoin': 'Offer.fk_primary_refusal_reason_code==RefusalReason.code',
                                'lazy': 'joined'})
    secondary_refusal_reason: Optional["RefusalReason"] = Relationship(
        sa_relationship_kwargs={'primaryjoin': 'Offer.fk_secondary_refusal_reason_code==RefusalReason.code',
                                'lazy': 'joined'})
