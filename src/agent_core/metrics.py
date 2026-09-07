from contextvars import ContextVar
from functools import wraps
import time

current_metrics = ContextVar("cat_agent_metrics", default=None)


def measured(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            started = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                metrics = current_metrics.get()
                if metrics is not None:
                    metrics[name] = metrics.get(name, 0.0) + time.monotonic() - started
        return call
    return decorate
