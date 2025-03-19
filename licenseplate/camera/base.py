from abc import ABC, abstractmethod

from numpy.typing import NDArray


class CameraInterface(ABC):

    @abstractmethod
    def get_frame(self) -> NDArray:
        pass
