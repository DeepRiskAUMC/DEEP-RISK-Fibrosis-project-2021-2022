from functools import partial
from .stage_net import network_factory

def get_model(cfg, *args, **kwargs):
    return network_factory(cfg, **kwargs)(cfg, pre_weights=cfg.PRE_WEIGHTS_PATH, **kwargs)
    #net = partial(network_factory(cfg), config=cfg, pre_weights=cfg.PRE_WEIGHTS_PATH)
    #return net(*args, **kwargs)
