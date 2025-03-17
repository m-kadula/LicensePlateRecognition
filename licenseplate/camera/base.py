from abc import ABC, abstractmethod

from numpy.typing import NDArray


class CameraInterface(ABC):

    @abstractmethod
    def initiate(self) -> None:
        pass

    @abstractmethod
    def get_frame(self) -> NDArray:
        pass

    @abstractmethod
    def deactivate(self) -> None:
        pass
