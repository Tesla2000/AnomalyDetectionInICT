import random
from functools import wraps
from typing import Callable

import numpy as np

from Config import Config


def _set_seed(function: Callable):
    @wraps(function)
    def wrapper(*args, **kwargs):
        random.seed(Config.random_state)
        np.random.seed(Config.random_state)
        return function(*args, **kwargs)

    return wrapper
