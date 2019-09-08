from gym.core import ObservationWrapper


class Binarizer(ObservationWrapper):
    def __init__(self, env, func) -> None:
        super().__init__(env)
        self._func = func

    def observation(self, state):
        return self._func(state)