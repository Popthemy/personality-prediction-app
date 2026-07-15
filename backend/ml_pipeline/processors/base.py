""""
process() — abstract method. Every processor subclass must implement this. It takes a CleanedContent object and returns a plain feature dict. The ABC enforcement means if you forget to implement it, Python raises a TypeError at instantiation rather than silently failing at runtime.
_safe_float() — coerces any value to float, returning a default (0.0) instead of crashing if the value is None, a non-numeric string, or otherwise uncastable. Used heavily in engagement and temporal processors where API fields can be missing.
_safe_int() — same pattern for integers. Used in profile and network processors for counts like followers_count that the API sometimes omits or returns as strings.
"""


from abc import ABC, abstractmethod
from typing import Any
from ..cleaning.cleaner import CleanedContent


class BaseProcessor(ABC):
    @abstractmethod
    def process(self, content: CleanedContent) -> dict[str, Any]:
        """
        Extract features from a CleanedContent item.
        Returns a dict where all values are JSON-serialisable
        (float, int, bool, str, or None).
        """

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
