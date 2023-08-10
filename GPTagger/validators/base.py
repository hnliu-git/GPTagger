from abc import ABC, abstractmethod
from typing import Any


class BaseValidator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, text: str) -> bool:
        pass
