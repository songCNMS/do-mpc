from typing import Any, List, Sequence, Tuple, Union
import sys
import numpy as np
from d3rlpy.argument_utility import ActionScalerArg
from d3rlpy.constants import ActionSpace
from d3rlpy.algos.base import AlgoBase
from examples.batch_reactor.template_mpc import template_mpc


class MPCPolicy(AlgoBase):
    _action_size: int

    def __init__(
        self,
        *,
        mpc_model=None,
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
        # x = np.asarray(x).flatten()
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
            
            if self.noise > 0:
                random_action = [np.random.random()*(a_max-a_min) + a_min for a_min, a_max in zip(self.min_actions, self.max_actions)]
                action = np.array([self.noise*r_a+(1.0-self.noise)*a for r_a, a in zip(random_action, action)])
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
    
    
    
def get_mpc_controller(env):
    mpc = template_mpc(env.model)
    mpc.x0 = env.state
    mpc.set_initial_guess()
    
    policy = MPCPolicy(mpc_model=mpc, 
                       noise=0.1, 
                       min_actions=env.min_actions, 
                       max_actions=env.max_actions)
    return policy