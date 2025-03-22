from typing import Self
from abc import ABC, abstractmethod
from datetime import datetime

from numpy.typing import NDArray

from ..detection import FinderResult, ExtractorResult


class ActionInterface(ABC):

    @classmethod
    def get_instance(cls, **kwargs) -> Self:
        return cls()

    @abstractmethod
    def action_if_found(
        self,
        image: NDArray,
        detected_plated: list[tuple[FinderResult, list[ExtractorResult]]],
        time: datetime,
    ):
        pass

    def action_if_not_found(self, image: NDArray, time: datetime):
        pass
