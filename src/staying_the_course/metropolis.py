"""Metropolis sampler module.

"""

import itertools
import math

from typing import Callable, Any

import numpy as np


class Metropolis:
    """Implementation of the Metropolis-Hastings algorithm.

    """
    def __init__(self, potential: Callable[[np.ndarray, Any], float],
                 temperature: float,
                 delta: float,
                 initial_point: np.ndarray) -> None:
        self.potential = potential
        self.temperature = temperature

        self.delta = delta

        self.original_initial_point = initial_point.copy()
        self.dim = initial_point.shape[0]

        self.reset()

    def reset(self) -> None:
        self.initial_point = self.original_initial_point.copy()
        self.current_point = self.initial_point.copy()
        self.probability = self._compute_probability(self.initial_point)

        self.accepted = self.total = 1

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(potential={self.potential}, '
                + f'temperature={self.temperature}, delta={self.delta}, '
                + f'initial_point={self.current_point})')

    def __iter__(self) -> 'Metropolis':
        return self

    def __next__(self) -> np.ndarray:
        candidate = self._generate_candidate()

        p_old = self.probability
        p_new = self._compute_probability(candidate)

        if p_new >= p_old or np.random.rand() * p_old < p_new:
            self.current_point = candidate
            self.probability = p_new
            self.accepted += 1

        self.total += 1
        return self.current_point

    def _generate_candidate(self) -> np.ndarray:
        r = self.delta * (2 * np.random.rand(self.dim)
                          - np.ones(self.dim))
        return self.current_point + r

    def get_acceptance_ratio(self) -> float:
        """Return current acceptance ratio.

        """
        return self.accepted / self.total

    def _compute_probability(self, point: np.ndarray) -> float:
        return math.exp(-self.potential(point) / self.temperature)

    def draw_samples(self, num_samples: int, step: int = 1) -> np.ndarray:
        """Sample a number of points from the chain.

        """
        points = np.zeros((num_samples // step, self.dim))
        iterator = itertools.islice(self, 0, num_samples, step)

        for i, point in enumerate(iterator):
            points[i, :] = point

        return points

    def ensemble_average(self, observable: Callable[[np.ndarray], float],
                         num_samples: int, step: int) -> float:
        """Evaluate the ensemble average of an observable.

        """
        average = 0.0
        total_samples = num_samples / step

        iterator = itertools.islice(self, 0, num_samples, step)
        for point in iterator:
            average += observable(point)

        return average / total_samples
