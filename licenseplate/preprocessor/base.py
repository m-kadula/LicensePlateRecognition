from typing import Self
from abc import ABC, abstractmethod

from numpy.typing import NDArray


class PreprocessorInterface(ABC):
    @classmethod
    def get_instance(cls, **kwargs) -> Self:
        return cls()

    @abstractmethod
    def preprocess(self, image: NDArray) -> NDArray:
        pass

    def __call__(self, image: NDArray) -> NDArray:
        return self.preprocess(image)


class IdentityPreprocessor(PreprocessorInterface):
    def preprocess(self, image: NDArray) -> NDArray:
        return image.copy()
