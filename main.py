import sys
import os
from collections.abc import Mapping
from copy import deepcopy
import numpy as np
import torch
import yaml
import pdb
import wandb
import warnings
import ast
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from Environment.utils.arguments import parse_args
from tqdm import tqdm

import yaml
from MARL.run import REGISTRY as run_REGISTRY
from MARL.utils.logging import get_logger

SETTINGS["CAPTURE_MODE"] = (
    "fd"  # set to "no" if you want to see stdout/stderr in console
)
logger = get_logger()
ex = Experiment("ctraffic")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


def _get_config(config_file):
    if config_file is not None:
        with open(
            config_file,
            "r",
        ) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{} error: {}".format(config_file, exc)
        return config_dict
    else:
        return {}


@ex.main
def main(_run, _config, _log):
    print(_log)
    config = deepcopy(_config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["enable_wandb"]:
        wandb.login()
        wandb.init(
            project=config["wandb_project_name"],
            entity=config["wandb_entity_name"],
            name=config["name"],
            config=config
        )
    run_REGISTRY[_config["config"]["run"]](_run, config, _log)


if __name__ == "__main__":
    args = sys.argv
    if any("--config" in arg for arg in args):
        index = -1
        for i, arg in enumerate(args):
            if "--config" in arg:
                index = i
                break
        alg_config = _get_config(args[index].split("=")[1])
        name = os.path.basename(args[index].split("=")[1]).split(".")[0]
        args = args[:index] + args[index + 1 :]
    for arg in args:
        if "--cityflow-config" in arg:
            cityflow_name = arg.split("=")[1]
            cityflow_name = cityflow_name[
                cityflow_name.index(".") - 2 : cityflow_name.index(".")
            ]
    command = args[0]
    args = vars(parse_args(args))
    args["config"] = alg_config
    args["name"] = name
    args["results_file"] = os.path.join(
        args["results_path"],
        f"{args['name']}_{cityflow_name}_{args['constraint']}_{args['seed']}_{args['ablation']}.pkl",
    )
    ex.add_config(args)
    file_obs_path = os.path.join(args["results_path"], "sacred")
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(command)
