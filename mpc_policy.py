from typing import Any, List, Sequence, Tuple, Union
import sys
import numpy as np
from d3rlpy.argument_utility import ActionScalerArg
from d3rlpy.constants import ActionSpace
from d3rlpy.algos.base import AlgoBase


class MPCPolicy(AlgoBase):
    _action_size: int

    def __init__(
        self,
        *,
        mpc_model=None,
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
        self._action_size = 1
        self._impl = None

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._action_size = action_size

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return self.sample_action(x)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        # Create the in-memory "file"
        action_list = []
        for obs in x:
            try:
                save_stdout = sys.stdout
                sys.stdout = open('.trash.log', 'w')
                ori_action = self.mpc_model.make_step(obs)
                action = ori_action.flatten()
            finally:
                sys.stdout = save_stdout
            action_list.append(action)
        return np.asarray(action_list)

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
    

class NoisyPolicy(AlgoBase):
    _action_size: int

    def __init__(
        self,
        *,
        base_model=None,
        noise=None,
        min_actions=None,
        max_actions=None,
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
        self.base_model = base_model
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
        # Create the in-memory "file"
        if self.noise >= 1.0:
            action_list = [[np.random.random()*(a_max-a_min) + a_min for a_min, a_max in zip(self.min_actions, self.max_actions)] for _ in x]
        else:
            original_actions = self.base_model.predict(x)
            action_list = []
            for ori_action in original_actions:
                if self.noise > 0:
                    random_action = [np.random.random()*(a_max-a_min) + a_min for a_min, a_max in zip(self.min_actions, self.max_actions)]
                    action = np.array([self.noise*r_a+(1.0-self.noise)*a for r_a, a in zip(random_action, ori_action)])
                else: action = ori_action
                action_list.append(action)
        return np.asarray(action_list)

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS  

    
    
def get_mpc_controller(env, noise=0.0):
    mpc = env.mpc
    env.reset()
    mpc.set_initial_guess()
    policy =  NoisyPolicy(base_model=MPCPolicy(mpc_model=mpc), 
                          noise=noise, 
                          min_actions=env.min_actions, 
                          max_actions=env.max_actions)
    return policy


def get_noisy_rl_policy(env, base_policy, noise=0.0):
    policy =  NoisyPolicy(base_model=base_policy, 
                          noise=noise, 
                          min_actions=env.min_actions, 
                          max_actions=env.max_actions)
    return policy