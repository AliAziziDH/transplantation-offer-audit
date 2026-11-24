from typing import TYPE_CHECKING, Optional

from sqlalchemy import Column, Enum
from sqlmodel import Field, Relationship, SQLModel

from src.models.enums import Answer, ConditionDescription

if TYPE_CHECKING:
    from src.models.recipient import Recipient


class Kidney(SQLModel, table=True):
    pump: Answer = Field(sa_column=Column(Enum(Answer)))
    interstitial_fibrosis_description: ConditionDescription | None = Field(
        default=None, sa_column=Column(Enum(ConditionDescription)))
    vascular_changes_description: ConditionDescription | None = Field(
        default=None, sa_column=Column(Enum(ConditionDescription)))

    kidney_id: int | None = Field(default=None, primary_key=True)

    fk_recipient_id: int | None = Field(default=None, foreign_key='recipient.recipient_id')

    recipient: Optional["Recipient"] = Relationship(
        back_populates='received_kidney', sa_relationship_kwargs={'post_update': True})
