import torch as th
import torch.nn as nn

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from gymnasium import spaces

from collections import namedtuple, OrderedDict


NormInfo = namedtuple('NormInfo', ['mean', 'std', 'clip'])


class WrapperNet(nn.Module) :
    '''
    - only continuous observation is supported
    - observation normalization is supported
    - reward normalization is not supported
    '''

    def __init__(self, model:BaseAlgorithm, env:VecEnv, wrapped_vecnormalize) :
        super(WrapperNet, self).__init__()
        self.net = model.policy
        
        # When model output doesn't have any relation with input, error occure.
        # This problem can be solve by register output tensor to nn.Parameter
        self.version = nn.Parameter(th.tensor([3], dtype=th.float32), False)
        self.memory_size = nn.Parameter(th.tensor([0], dtype=th.float32), False)
        self.output_shape = nn.Parameter(th.tensor(model.action_space.shape, dtype=th.float32), False)

        self.obs_names = [key for key in env.observation_space.spaces]

        # normalize
        self.norm_obs = wrapped_vecnormalize and env.norm_obs
        self.dict_obs = isinstance(env.observation_space, spaces.Dict)

        if self.norm_obs :
            if self.dict_obs :
                obs_mean = []
                obs_std = []
                for _, obs_rms in env.obs_rms.items() :
                    obs_mean.append(th.from_numpy(obs_rms.mean))
                    obs_std.append(th.sqrt(th.from_numpy(obs_rms.var) + env.epsilon))
            else :
                obs_mean = th.from_numpy(env.obs_rms.mean)
                obs_std = (th.from_numpy(env.obs_rms.var) + env.epsilon).sqrt()
            
            self.norm_obs_info = NormInfo(
                mean = obs_mean,
                std = obs_std,
                clip = env.clip_obs
            )
    
    def forward(self, *args) :
        # normalize
        if self.norm_obs :
            if self.dict_obs :
                obs = OrderedDict([
                    (
                        name,
                        th.clip(
                            input=(args[idx] - self.norm_obs_info.mean[idx]) / self.norm_obs_info.std[idx],
                            min=-self.norm_obs_info.clip,
                            max=self.norm_obs_info.clip
                        )
                    ) for idx, name in enumerate(self.obs_names)
                ])
            else :
                obs = th.clip(
                    input=(args[0] - self.norm_obs_info.mean) / self.norm_obs_info.std,
                    min=-self.norm_obs_info.clip,
                    max=self.norm_obs_info.clip
                )
        
        # sample actions
        actions, _, _ = self.net(obs, deterministic=True)
        
        return self.version, self.memory_size, actions, self.output_shape


def export_as_onnx(model:BaseAlgorithm, path:str, env:VecEnv, wrapped_vecnormalize:bool=False) :
    # model input sample
    dummy_input = tuple(
        th.rand((1,)+box_space.shape, dtype=th.float32)
        for name, box_space in env.observation_space.spaces.items()
    )
    
    # export as onnx
    wrapped_net = WrapperNet(model, env, wrapped_vecnormalize)

    th.onnx.export(
        model=wrapped_net,
        args=dummy_input,
        f=path,
        opset_version=11,
        
        input_names=wrapped_net.obs_names,
        output_names=[
            'version_number',
            'memory_size',
            'continuous_actions',
            'continuous_action_output_shape'
        ],
        dynamic_axes={
            **{key : {0: 'batch'} for key in env.observation_space.spaces},
            **{
                'continuous_actions' : {0: 'batch'}
            }
        }
    )