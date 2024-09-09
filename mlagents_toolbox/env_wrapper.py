from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from gymnasium import spaces


class ActionBoundScaler(VecEnvWrapper):
    def __init__(self, env: VecEnv, action_bound:float=3.0):
        self.action_bound = action_bound
        super().__init__(
            venv=env,
            observation_space=env.observation_space,
            action_space=spaces.Box(
                env.action_space.low  * self.action_bound,
                env.action_space.high * self.action_bound,
                env.action_space.shape,
                env.action_space.dtype
            )
        )
    
    def reset(self):
        obs = self.venv.reset()
        return obs
    
    def step_async(self, actions):
        self.venv.step_async(actions / self.action_bound)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info