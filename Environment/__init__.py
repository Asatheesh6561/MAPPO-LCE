import os
from Environment.cityflow_env import CityFlowEnv
from Environment.wrapper.default_wrapper import DefaultWrapper
from Environment.wrapper.oracle_wrapper import OracleWrapper
from Environment.wrapper.flatten_wrapper import FlattenWrapper
from Environment.wrapper.observation_wrapper import ObsWrapper


def make_env(config, wrapper_names=["DefaultWrapper"]):
    log_folder = config["log_folder"]
    work_folder = config["work_folder"]
    cityflow_config = config["cityflow_config"]
    seed = config["seed"]
    env = CityFlowEnv(log_folder, work_folder, cityflow_config, seed=seed)
    for wrapper_name in wrapper_names:
        env = eval(wrapper_name)(env)
    return env
