from enum import Enum, auto

class PipelineState(Enum):
    IDLE = auto()
    TRIGGERED = auto()
    CAPTURING = auto()
    PROCESSING = auto()
    ERROR = auto()

# Internal mutable state
_current_state = PipelineState.IDLE

def set_state(new_state: PipelineState) -> bool:
    """Attempt to transition to a new state.
    Returns True if transition succeeded, False if invalid.
    """
    global _current_state
    valid_transitions = {
        PipelineState.IDLE: [PipelineState.TRIGGERED],
        PipelineState.TRIGGERED: [PipelineState.CAPTURING, PipelineState.ERROR],
        PipelineState.CAPTURING: [PipelineState.PROCESSING, PipelineState.ERROR],
        PipelineState.PROCESSING: [PipelineState.IDLE, PipelineState.ERROR],
        PipelineState.ERROR: [PipelineState.IDLE],
    }
    if new_state not in valid_transitions.get(_current_state, []):
        print(f"[STATE] Invalid transition {_current_state} -> {new_state}; ignoring.")
        return False
    print(f"[STATE] Transition {_current_state} -> {new_state}")
    _current_state = new_state
    return True

def get_state() -> PipelineState:
    return _current_state

def reset_state():
    """Force reset to IDLE (used in finally blocks)."""
    global _current_state
    _current_state = PipelineState.IDLE
    print("[STATE] Reset to IDLE")
