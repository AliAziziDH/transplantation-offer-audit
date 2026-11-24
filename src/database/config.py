from os import getenv

from sqlmodel import Session, SQLModel, create_engine


class DBManager:
    DEFAULT_NAME = 'transplantation'
    DEFAULT_DBMS = 'postgresql'
    DEFAULT_HOST = '127.0.0.1'
    DEFAULT_PORT = '5432'

    _instance = None
    _engine = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton so tests or CLI runs can reconfigure environment variables."""
        if cls._engine is not None:
            cls._engine.dispose()
        cls._instance = None
        cls._engine = None

    def _initialize(self):
        self.db_uri = getenv('DB_URI')
        self.dbms = getenv('DB_DBMS') or self.DEFAULT_DBMS
        self.host = getenv('DB_HOST') or self.DEFAULT_HOST
        self.port = getenv('DB_PORT') or self.DEFAULT_PORT
        self.name = getenv('DB_NAME') or self.DEFAULT_NAME
        self.user = getenv('DB_USER')
        self.password = getenv('DB_PASSWORD')
        echo = getenv('DB_ECHO', 'false').lower() == 'true'
        self._engine = create_engine(self.get_uri(), echo=echo)

    def get_uri(self) -> str:
        if self.db_uri:
            return self.db_uri
        if self.dbms == 'sqlite':
            if not self.name or self.name == ':memory:':
                return 'sqlite:///:memory:'
            return f'sqlite:///{self.name}'
        return f'{self.dbms}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}'

    def get_engine(self):
        return self._engine

    def get_db(self) -> Session:
        return Session(self.get_engine())

    def create_db_and_tables(self):
        SQLModel.metadata.create_all(self._engine)
