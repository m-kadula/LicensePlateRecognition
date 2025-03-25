from typing import Self, Any
from abc import ABC, abstractmethod

from numpy.typing import NDArray


class CameraInterface(ABC):
    @classmethod
    def get_instance(cls, kwargs: dict[str, Any]) -> Self:
        return cls()

    @abstractmethod
    def get_frame(self) -> NDArray:
        pass
