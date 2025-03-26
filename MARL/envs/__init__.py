from functools import partial
from .multiagentenv import MultiAgentEnv
from .traffic.CTrafficEnvironment import CTrafficEnvironment


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

REGISTRY["ctraffic"] = partial(env_fn, env=CTrafficEnvironment)
