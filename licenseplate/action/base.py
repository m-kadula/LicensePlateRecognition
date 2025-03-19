from abc import ABC, abstractmethod

from numpy.typing import NDArray

from ..detection import FinderResult, ExtractorResult


class ActionInterface(ABC):

    @abstractmethod
    def action_if_found(self, image: NDArray, detected_plated: list[tuple[FinderResult, list[ExtractorResult]]]) -> bool:
        pass
