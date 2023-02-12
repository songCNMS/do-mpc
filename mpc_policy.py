from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from d3rlpy.argument_utility import ActionScalerArg
from d3rlpy.constants import ActionSpace
from d3rlpy.algos.base import AlgoBase


class MPCPolicy(AlgoBase):
    _action_size: int

    def __init__(
        self,
        *,
        mpc_model,
        noise,
        min_actions,
        max_actions,
        action_scaler: ActionScalerArg = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=1,
            n_frames=1,
            n_steps=1,
            gamma=0.0,
            scaler=None,
            action_scaler=action_scaler,
            kwargs=kwargs,
        )
        self.mpc_model = mpc_model
        self.noise = noise
        self.min_actions = min_actions
        self.max_actions = max_actions
        self._action_size = 1
        self._impl = None

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._action_size = action_size

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return self.sample_action(x)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        x = np.asarray(x)
        action = self.mpc_model.make_step(x)
        if self.noise > 0:
            random_action = [np.random.random()*(a_max-a_min) + a_min for a_min, a_max in zip(self.min_actions, self.max_actions)]
            action = np.array([self.noise*r_a+(1.0-self.noise)*a for r_a, a in zip(random_action, action)])
        return action

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS