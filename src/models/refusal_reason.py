from sqlmodel import Field, SQLModel


class RefusalReason(SQLModel, table=True):

    __tablename__ = 'refusal_reason'

    code: int | None = Field(default=None, primary_key=True)
    description: str
