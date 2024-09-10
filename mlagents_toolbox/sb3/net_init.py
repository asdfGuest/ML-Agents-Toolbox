import torch as th
from enum import Enum


class Initialization(Enum) :
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3
    KaimingHeUniform = 4
    Normal = 5


_init_methods = {
    Initialization.Zero: th.nn.init.zeros_,
    Initialization.XavierGlorotNormal: th.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: th.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: th.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: th.nn.init.kaiming_uniform_,
    Initialization.Normal: th.nn.init.normal_,
}


def _init_linear_layer(layer:th.nn.Linear, init_type:Initialization, gain:float=1.0) :

    is_kaiming = init_type == Initialization.KaimingHeNormal or init_type == Initialization.KaimingHeUniform
    init_kwargs = dict(nonlinearity='linear') if is_kaiming else dict()    

    # weight initialization
    _init_methods[init_type](
        layer.weight.data,
        **init_kwargs
    )
    layer.weight.data *= gain
    # bias initialization
    _init_methods[Initialization.Zero](layer.bias.data)


def _find_init_linear_layer(layers, init_type:Initialization, gain:float=1.0) :
    cnt = 0
    for layer in layers :
        if isinstance(layer, th.nn.Linear) :
            _init_linear_layer(layer, init_type, gain)
            cnt += 1    
    return cnt


def init_network(model) :
    cnt = 0
    
    cnt += _find_init_linear_layer(
        model.policy.mlp_extractor.policy_net,
        Initialization.KaimingHeNormal,
        gain=1.0
    )
    cnt += _find_init_linear_layer(
        model.policy.mlp_extractor.value_net,
        Initialization.KaimingHeNormal,
        gain=1.0
    )

    cnt += _find_init_linear_layer(
        model.policy.action_net,
        Initialization.KaimingHeNormal,
        gain=0.2
    )
    cnt += _find_init_linear_layer(
        model.policy.value_net,
        Initialization.XavierGlorotUniform,
        gain=1.0
    )

    print('Total %d layers were initializaed.'%(cnt,))