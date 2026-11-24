from enum import Enum, auto


class BirthSex(Enum):
    MALE = auto()    # Male
    FEMALE = auto()  # Female


class Answer(Enum):
    YES = auto()              # Y
    NO = auto()               # N
    PROVISIONAL_YES = auto()  # Z
    UNCERTAIN = auto()        # U


class GraftStatus(Enum):
    FUNCTIONING = auto()
    FAILED = auto()
    UNKNOWN = auto()


class PatientStatus(Enum):
    ALIVE = auto()
    DECEASED = auto()
    UNKNOWN = auto()


class BloodType(Enum):
    O = auto()
    A = auto()
    B = auto()
    AB = auto()
    UNKNOWN = auto()


class CauseOfDeath(Enum):
    CEREBROVASCULAR_STROKE = auto()
    ANOXIA = auto()
    HEAD_TRAUMA = auto()
    TUMOR = auto()
    OTHER = auto()


class ConditionDescription(Enum):
    ABSENT = auto()
    MILD_MODERATE = auto()
    MILD = auto()
    MINIMAL = auto()
    NONE = auto()
    UNKNOWN = auto()


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()


class RefusalReasonType(Enum):
    PRIMARY = auto()
    SECONDARY = auto()
