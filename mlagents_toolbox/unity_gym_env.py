from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvIndices

import numpy as np
import gymnasium as gym

from collections import OrderedDict
from typing import List, Tuple, Type, Any, Iterable


def _raise_error(description:str, trigger:bool):
    if trigger :
        raise Exception(description)

class _CHECKER() :
    BEHAVIOR_NUM = lambda behavior_num : _raise_error(
        'Behavior number in scene is %d. It must be one.'%behavior_num,
        behavior_num != 1
    )
    BEHAVIOR_NAME = lambda current_name, expected_name : _raise_error(
        'Behavior name is not same with expected one.\n'
        'current behavior name : %s   expected behavior name : %s'%(current_name, expected_name),
        current_name != expected_name
    )
    AGENT_ID = lambda current_agents_id, expecteds_agent_id : _raise_error(
        'Behavior name is not same with expected one.\n' +
        'current agents id : %s\n'%current_agents_id +
        'expected agents id : %s\n'%expecteds_agent_id,
        current_agents_id != expecteds_agent_id
    )
    OBSERVATION_SPACE = lambda obs_specs : _raise_error(
        'Only vector observation is allowed.',
        any([len(obs_spec.shape) != 1 for obs_spec in obs_specs])
    )


def get_env_info(unity_env:UnityEnvironment) -> Tuple[str, List[int], gym.spaces.Space, gym.spaces.Space] :
    '''
    you need to call unity_env.reset() before call this function
    '''

    _CHECKER.BEHAVIOR_NUM(len(unity_env.behavior_specs))
    behavior_name = list(unity_env.behavior_specs.keys())[0]

    decision_steps, _ = unity_env.get_steps(behavior_name)
    agents_id = decision_steps.agent_id.tolist()

    # observation space
    behavior_spec = unity_env.behavior_specs[behavior_name]
    _CHECKER.OBSERVATION_SPACE(behavior_spec.observation_specs)
    
    obs_space = []
    for idx, obs_spec in enumerate(behavior_spec.observation_specs) :
        bound = np.full(obs_spec.shape, fill_value=np.inf, dtype=np.float32)
        
        obs_space.append((
            'obs_%d'%idx,
            gym.spaces.Box(
                low=-bound,
                high=bound,
                shape=obs_spec.shape,
                dtype=np.float32
            )
        ))
    obs_space = gym.spaces.Dict(obs_space)

    # action space
    act_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(behavior_spec.action_spec.continuous_size,),
        dtype=np.float32
    )

    return behavior_name, agents_id, obs_space, act_space


class UnityVecEnv(VecEnv) :
    '''

    Wrapper for unity multi-instance environment.
    This wrapper will wrap UnityEnvironment to VecEnv.

    Not every unity environments can be wrapped.
    See the following restrictions :

    - only one kind of behavior is allowed in one scene
    - agents number in scene must not change
    - all agents in scene must have same decision request frequency

    - only continuous action is supported (we will support discrete action space in future)
    - only vector observation is supported (we will support other kind of observation in futer)

    '''
    
    def __init__(self, unity_env: UnityEnvironment) :
        # unity env
        self.unity_env = unity_env

        # environment information
        self.unity_env.reset()
        behavior_name, agents_id, obs_space, act_space = get_env_info(self.unity_env)

        self.behavior_name = behavior_name
        self.agents_id = agents_id
        self.obs_num = len(obs_space)
        
        super().__init__(
            num_envs = len(self.agents_id),
            observation_space = obs_space,
            action_space = act_space
        )
        
        # data
        self.actions = None
        self.attributs = [{'render_mode':None} for _ in range(self.num_envs)]
    
    
    def __str__(self) :
        return (
            '---------- Unity Environments Information ----------\n' +
            'behavior name : %s\n'%self.behavior_name +
            'agents num : %d\n'%len(self.agents_id) +
            'agents id : %s\n'%str(self.agents_id) +
            'observation num : %d\n'%self.obs_num +
            'observation space : %s\n'%str(self.observation_space) +
            'action space : %s\n'%str(self.action_space) +
            '----------------------------------------------------'
        )
    

    def reset(self) -> VecEnvObs:
        self.unity_env.reset()

        _CHECKER.BEHAVIOR_NUM(len(self.unity_env.behavior_specs))
        _CHECKER.BEHAVIOR_NAME(list(self.unity_env.behavior_specs.keys())[0], self.behavior_name)
        
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        return self._obs2dict(decision_steps.obs)
    
    
    def step_async(self, actions: np.ndarray):
        self.actions = ActionTuple(continuous=actions)
    

    def step_wait(self) -> VecEnvStepReturn:
        # set actions
        self.unity_env.set_actions(self.behavior_name, self.actions)

        done = np.zeros((self.num_envs,), dtype=bool)
        info = [{'TimeLimit.truncated':False} for _ in range(self.num_envs)]

        # do actions
        while True :
            self.unity_env.step()
            decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
            
            # terminal steps
            for k in range(terminal_steps.obs[0].shape[0]) :
                agent_idx = self.agents_id.index(terminal_steps.agent_id[k])
                done[agent_idx] = True
                info[agent_idx]['TimeLimit.truncated'] = terminal_steps.interrupted[k]
                info[agent_idx]['terminal_observation'] = self._obs2dict([obs[k] for obs in terminal_steps.obs])
            
            if decision_steps.obs[0].shape[0] > 0 :
                break
        
        # decision steps
        _CHECKER.AGENT_ID(decision_steps.agent_id.tolist(), self.agents_id)

        obs = self._obs2dict(decision_steps.obs)
        reward = decision_steps.reward
        
        return obs, reward, done, info
    
    
    def close(self) -> None:
        self.unity_env.close()

    
    def _obs2dict(self, obs:List[np.ndarray]) :
        return OrderedDict([('obs_%d'%k, obs[k]) for k in range(len(obs))])


    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return [getattr(self.attributs[i], attr_name) for i in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass
    
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in self._get_indices(indices)]