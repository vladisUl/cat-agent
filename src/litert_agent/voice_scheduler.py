"""Compatibility import; CORE lives in agent_core."""
import sys
from agent_core import voice_scheduler as _implementation
sys.modules[__name__] = _implementation
