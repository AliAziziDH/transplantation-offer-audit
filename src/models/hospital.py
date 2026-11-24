from sqlmodel import Field, SQLModel


class Hospital(SQLModel, table=True):

    code: str | None = Field(default=None, primary_key=True)
    name: str
