from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast


class TRPOConfig(object):
    def __init__(
        self
    ) -> None:
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
 