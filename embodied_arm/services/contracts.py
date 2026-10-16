"""Public product contracts. These do not authorize network or robot actions."""
from dataclasses import dataclass
from enum import Enum

class TaskState(str, Enum):
    IDLE = 'idle'
    QUEUED = 'queued'
    OBSERVING = 'observing'
    PLANNING = 'planning'
    EXECUTING = 'executing'
    COMPLETED = 'completed'
    BLOCKED = 'blocked'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

@dataclass(frozen=True)
class TaskBudget:
    model_calls: int = 12
    vision_calls: int = 6
    seconds: int = 180

    def __post_init__(self):
        if any((type(v) is not int for v in (self.model_calls, self.vision_calls, self.seconds))):
            raise ValueError('Integer task budget required')
        if not 1 <= self.model_calls <= 200 or not 1 <= self.vision_calls <= 12 or (not 1 <= self.seconds <= 420):
            raise ValueError('Task budget outside supported bounds')

@dataclass(frozen=True)
class TaskIntent:
    action: str
    object_id: str | None = None
    target_id: str | None = None

    def __post_init__(self):
        if self.action not in ('pick', 'place', 'status', 'stop', 'clarify'):
            raise ValueError('Unsupported product intent')
        if self.action in ('pick', 'place') and self.object_id not in ('red', 'blue'):
            raise ValueError('Unknown object')
        if self.action == 'place' and self.target_id != 'goal':
            raise ValueError('Single-block placement supports goal only')
        if self.action == 'pick' and self.target_id is not None:
            raise ValueError('Pick holds the object')
        if self.action in ('status', 'stop', 'clarify') and (self.object_id is not None or self.target_id is not None):
            raise ValueError('Conversation cannot carry motion targets')
