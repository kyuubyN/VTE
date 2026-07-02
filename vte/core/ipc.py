from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class UIMsgPrompt:
    text: str
    max_tokens: int = 512

@dataclass
class UIMsgCancel:
    pass

@dataclass
class UIMsgShutdown:
    pass

@dataclass
class MotorMsgToken:

    text: str

@dataclass
class MotorMsgMetrics:
    temp_c: float
    clock_mhz: float
    vram_mb: float
    power_w: float
    tokens_sec: float
    vram_free_system_mb: float = 0.0
    vram_details: Optional[dict] = None

@dataclass
class MotorMsgProgress:
    status: str
    percentage: float
    
@dataclass
class MotorMsgReady:
    pass

@dataclass
class MotorMsgError:
    message: str

@dataclass
class MotorMsgStatusUpdate:
    is_loaded: bool
    time_until_unload: Optional[float]
