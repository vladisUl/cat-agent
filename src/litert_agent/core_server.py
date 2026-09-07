"""Compatibility import; CORE lives in agent_core."""
import sys
from agent_core import core_server as _implementation
sys.modules[__name__] = _implementation
