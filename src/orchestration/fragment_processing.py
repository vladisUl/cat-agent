"""Shared lifecycle and cooperative boundaries for isolated fragment inference."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging

LOGGER = logging.getLogger(__name__)
fragment_observer = ContextVar("fragment_observer", default=None)


def finish_steps(steps):
    """Synchronous compatibility entry point; CORE consumes the iterator instead."""
    try:
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value
    finally:
        steps.close()


def fork_child(client, label):
    fork = getattr(client, "fork", None)
    if not callable(fork):
        raise ValueError("model client does not support isolated contexts")
    try:
        return fork(label, inherit_base=False)
    except TypeError:
        return fork(label)


@contextmanager
def managed_child(child):
    try:
        yield child
    finally:
        close = getattr(child, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.exception("fragment child close failed")


def _progress(value):
    observer = fragment_observer.get()
    if observer is not None:
        observer(value)


def fragment_step(child, messages, base_messages, *, tool, index, total, source):
    """One inference, BASE reset, then a scheduler/cancellation boundary.

    The child's history never moves into another request. Generator.close()
    unwinds this boundary and the enclosing managed_child on cancellation.
    """
    progress = dict(tool=tool, index=index, total=total, source=source)
    try:
        _progress({**progress, "phase": "running"})
        response = child.chat(messages)
        result = response.content.strip()
        reset = getattr(child, "reset_to_base", None)
        if callable(reset):
            reset(base_messages)
        _progress({**progress, "phase": "between_fragments"})
        yield
        return result
    finally:
        _progress(None)
