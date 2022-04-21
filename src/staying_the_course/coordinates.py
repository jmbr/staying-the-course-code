from abc import ABC, abstractmethod


class Coordinates(ABC):
    domain_dimension: int
    codomain_dimension: int

    @abstractmethod
    def __call__(self, X):
        ...
