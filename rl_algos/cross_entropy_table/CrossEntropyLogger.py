from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast
import numpy as np
import matplotlib.pyplot as plt
from rl_algos.cross_entropy_table.CrossEntropyTable import Episode


class CrossEntropyLogger(object):
    def __init__(
        self,
        percentile: float,
    ) -> None:
        self._percentile = percentile

        self.mins: List[float] = []
        self.maxs: List[float] = []
        self.thresholds: List[float] = []
        self.avgs: List[float] = []

    def log(self, episodes: List[Episode]) -> None:
        rewards = list(map(lambda x: x.total_reward, episodes))

        self.mins.append(min(rewards))
        self.maxs.append(max(rewards))
        self.thresholds.append(np.percentile(rewards, self._percentile))
        self.avgs.append(sum(rewards) / len(rewards))

    def plot(self) -> None:
        x = list(range(len(self.mins)))
        #plt.plot(x, self.mins)
        #plt.plot(x, self.maxs)
        #plt.plot(x, self.thresholds)
        plt.plot(x, self.avgs)
        #plt.legend(['mins', 'maxs', "thresholds", 'avgs'])
        plt.show()
        print(sum(self.avgs) / len(self.avgs))