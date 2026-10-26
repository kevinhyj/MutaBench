"""Model-generation waits, user policy dated 2026-10-13.

Apply at the wire boundary, not by rewriting frozen requests or prompts.
Connection establishment, metadata GETs and scheduler polling are separate.
"""
import math

MIN_GENERATION_TIMEOUT_S = 3600

def generation_timeout_seconds(configured=None):
    """At least one hour; retain an already longer wait (e.g. 7200 s)."""
    if configured is None:
        return MIN_GENERATION_TIMEOUT_S
    if isinstance(configured, bool):
        raise ValueError('Generation timeout must be finite and positive')
    value = float(configured)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Generation timeout must be finite and positive')
    return max(MIN_GENERATION_TIMEOUT_S, value)
