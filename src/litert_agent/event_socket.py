"""Compatibility import; CORE lives in agent_core."""
import sys
from agent_core import event_socket as _implementation
sys.modules[__name__] = _implementation
