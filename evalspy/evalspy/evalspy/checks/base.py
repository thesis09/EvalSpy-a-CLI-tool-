from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Severity(Enum):
    CRITICAL = "CRITICAL"   # will produce wrong results
    WARNING  = "WARNING"    # may produce wrong results
    INFO     = "INFO"       # good to know


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name:        str
    status:      Status
    severity:    Severity
    description: str
    detail:      Optional[str] = None
    fix:         Optional[str] = None
    line:        Optional[int] = None
