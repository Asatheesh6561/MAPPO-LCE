import datetime
import glob
import os
import re
import threading
import time
import copy
from os.path import abspath, dirname
from types import SimpleNamespace as SN
import pandas as pd
import numpy as np
import torch
import pdb

import wandb
from MARL.components.episode_buffer import ReplayBuffer
from MARL.components.transforms import OneHot
from MARL.components.reward_scaler import RewardScaler
from MARL.controllers import REGISTRY as mac_REGISTRY
from MARL.learners import REGISTRY as le_REGISTRY
from MARL.runners import REGISTRY as r_REGISTRY

from MARL.utils.logging import Logger
from MARL.utils.timehelper import time_left, time_str
import pickle as pkl


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    # args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    tmp_config = {
        k: _config[k]
        for k in _config
        if type(_config[k]) != dict and type(_config[k]) != list
    }
    tmp_config.update({f"config.{k}": _config["config"][k] for k in _config["config"]})
    print(
        pd.Series(tmp_config, name="HyperParameter Value")
        .transpose()
        .sort_index()
        .fillna("")
        .to_markdown()
    )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.config["runner"]](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "mean_action": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "reward": {"vshape": (1,)},
        "costs": {"vshape": (1,)},
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "individual_costs": {"vshape": (1,), "group": "agents"},
        "total_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.config["buffer_size"],
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.config["buffer_cpu_only"] else args.device,
    )

    logger.console_logger.info("MDP Components:")
    print(pd.DataFrame(buffer.scheme).transpose().sort_index().fillna("").to_markdown())

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.config["mac"]](buffer.scheme, groups, args)

    val_args = copy.deepcopy(args)
    val_args.mode = "validation"
    val_runner = r_REGISTRY[args.config["runner"]](args=val_args, logger=logger)

    test_args = copy.deepcopy(args)
    test_args.mode = "test"
    test_runner = r_REGISTRY[args.config["runner"]](args=test_args, logger=logger)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    val_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    test_returns = []

    # Learner
    learner = le_REGISTRY[args.config["learner"]](mac, buffer.scheme, logger, args)

    # Reward scaler
    reward_scaler = RewardScaler()

    if args.use_cuda:
        learner.cuda()

    # Start training
    episode = 0
    last_test_T = 0
    last_log_T = 0
    model_save_time = 0
    visual_time = 0
    test_best_return = -np.inf
    log_dicts = []

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.config["t_max"])
    )

    # Pre-collect samples to fit reward scaler
    if args.config["use_reward_normalization"]:
        episode_batch, train_old_return, train_stats = runner.run(test_mode=False)
        reward_scaler.fit(episode_batch)

    while runner.t_env <= args.config["t_max"]:
        # Step 1: Collect samples
        with torch.no_grad():
            episode_batch, train_reward, train_stats = runner.run(test_mode=False)
            wandb_dict = {}
            wandb_dict.update(
                {
                    "Train Reward": np.mean(train_reward),
                }
            )
            for k, v in train_stats.items():
                wandb_dict.update({f"train_{k}": v})

            if args.config["use_reward_normalization"]:
                episode_batch = reward_scaler.transform(episode_batch)
            buffer.insert_episode_batch(episode_batch)

        # Step 2: Train
        if buffer.can_sample(args.config["batch_size"]):
            next_episode = episode + args.config["batch_size_run"]
            if (args.accumulated_episodes == None) or (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes == 0
            ):
                episode_sample = buffer.sample(args.config["batch_size"])

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if args.use_cuda and episode_sample.device == "cpu":
                    episode_sample.to("cuda")
                learner.train(episode_sample, runner.t_env, episode)

        # Step 3: Evaluate
        if runner.t_env >= last_test_T + args.config["test_interval"]:
            # print("test-------------------------")
            # Log to console
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.config["t_max"])
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(
                        last_time, last_test_T, runner.t_env, args.config["t_max"]
                    ),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()
            last_test_T = runner.t_env
            test_old_return, test_stats = test_runner.run(test_mode=True)
            test_old_return = np.mean(test_old_return)
            wandb_dict.update(
                {
                    "Test Reward": test_old_return,
                }
            )
            for k, v in test_stats.items():
                wandb_dict.update({f"test_{k}": v})
            test_returns.append(test_old_return)
            if test_old_return > test_best_return:
                test_best_return = test_old_return
                print("new test result : {}".format(test_old_return))

        # Step 4: Log
        if args.enable_wandb:
            wandb.log(wandb_dict, step=runner.t_env)
        wandb_dict.update({"Time Step": runner.t_env})
        log_dicts.append(wandb_dict)

        # Step 5: Finalize
        episode += args.config["batch_size_run"]
        if (runner.t_env - last_log_T) >= args.config["log_interval"]:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    with open(args.results_file, "wb") as f:
        pkl.dump(log_dicts, f)
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["config"]["test_nepisode"] < config["config"]["batch_size_run"]:
        config["config"]["test_nepisode"] = config["config"]["batch_size_run"]
    else:
        config["config"]["test_nepisode"] = (
            config["config"]["test_nepisode"] // config["config"]["batch_size_run"]
        ) * config["config"]["batch_size_run"]

    return config


def discrete_derivative(test_returns):
    test_returns = np.array(test_returns)
    test_returns = test_returns[1:] - test_returns[:-1]
    test_returns = np.abs(test_returns)
    return 100 * test_returns.std()
