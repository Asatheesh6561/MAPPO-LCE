import os
import pdb
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pipe, Process
import time
import numpy as np
import torch.nn.functional as F
import torch
import faulthandler
import wandb
from MARL.components.episode_buffer import EpisodeBatch
from MARL.envs import REGISTRY as env_REGISTRY
from MARL.utils.timehelper import TimeStat


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.config["batch_size_run"]
        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args for _ in range(self.batch_size)]

        for i in range(len(env_args)):
            env_args[i].seed += i

        self.ps = [
            Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, config=vars(env_arg))),
                ),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_rewards = []
        self.test_rewards = []
        self.train_stats = {}
        self.test_stats = {}

        # self.time_stats = defaultdict(lambda: TimeStat(1000))
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False, storage_capacity=None):
        self.batch = self.new_batch()

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

        self.train_returns = []
        self.test_returns = []
        self.train_rewards = []
        self.test_rewards = []

    def compute_reward(self, reward, env_info):
        if self.args.constraint == "PhaseSkip":
            reward -= env_info["phase_skips"] * self.args.phase_skip_penalty
        elif self.args.constraint == "GreenSkip":
            reward -= env_info["green_skips"] * self.args.green_skip_penalty
        elif self.args.constraint == "GreenTime":
            reward -= env_info["green_times"] * self.args.green_time_penalty
        else:
            pass
        return reward

    def run(
        self,
        lbda_index=None,
        test_mode=False,
    ):
        faulthandler.enable()
        self.reset(test_mode=test_mode)

        all_terminated = False
        episode_returns = np.zeros([self.batch_size, self.args.config["n_lambda"]])
        episode_lengths = [0 for _ in range(self.batch_size)]
        if self.args.config["use_n_lambda"]:
            episode_individual_returns = np.zeros(
                [
                    self.batch_size,
                    self.args.n_agents,
                    self.args.config["n_lambda"],
                ]
            )
        else:
            episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents])

        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = self.args.config.get("save_probs", False)

        while True:
            if (
                self.args.config["mac"] == "mappo_mac"
                or self.args.config["mac"] == "maddpg_mac"
            ):
                mac_output = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )
            elif (
                self.args.config["mac"] == "dqn_mac"
                or self.args.config["mac"] == "ldqn_mac"
            ):
                mac_output = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    lbda_indices=None,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )

            if save_probs:
                actions, probs = mac_output
            else:
                actions = mac_output

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }

            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu").detach()

            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each environment
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "total_reward": [],
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data["individual_rewards"].append(
                        data["info"]["individual_rewards"]
                    )
                    post_transition_data["total_reward"].append(
                        data["info"]["total_reward"]
                    )

                    episode_returns[idx] += data["reward"]

                    if self.args.n_agents > 1:
                        episode_individual_returns[idx] += self.compute_reward(
                            data["info"]["individual_rewards"], data["info"]
                        )
                    else:
                        episode_individual_returns[idx] += self.compute_reward(
                            data["info"]["individual_rewards"][0], data["info"]
                        )

                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["mean_action"].append(
                        F.one_hot(actions[idx], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get reward for each env
        episode_rewards = [
            sum(episode_individual_returns) / self.t * (self.episode_limit)
        ]

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_rewards = self.test_rewards if test_mode else self.train_rewards

        infos = [cur_stats] + final_env_infos

        # Get the current total number of episodes
        n_episodes = cur_stats.get("n_episodes", 0)
        new_episodes = len(final_env_infos)
        total_episodes = n_episodes + new_episodes

        # Update cur_stats with the new running average
        for k in set.union(*[set(d) for d in infos]):
            cur_value = (
                cur_stats.get(k, 0) * n_episodes
            )  # Total accumulated value for `k`
            new_value = sum(
                d.get(k, 0) for d in final_env_infos
            )  # Sum of new batch values for `k`
            cur_stats[k] = (cur_value + new_value) / total_episodes
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        cur_rewards.extend(episode_rewards)

        if test_mode:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]

            cur_rewards = np.array(cur_rewards)
            rewards = cur_rewards.mean()
            final_stats = {}
            for k in cur_stats:
                if k in [
                    "average_time",
                    "average_delay",
                    "throughput",
                    "average_wait_time",
                    "green_times",
                    "phase_skips",
                    "green_skips",
                ]:
                    final_stats[k] = cur_stats[k]
            return final_stats, lambda_return, rewards
        else:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]

            cur_rewards = np.array(cur_rewards)
            rewards = cur_rewards.mean()
            final_stats = {}
            for k in cur_stats:
                if k in [
                    "average_time",
                    "average_delay",
                    "throughput",
                    "average_wait_time",
                    "green_times",
                    "phase_skips",
                    "green_skips",
                ]:
                    final_stats[k] = cur_stats[k]
            return self.batch, final_stats, lambda_return, rewards

    def _log(self, returns, individual_returns, rewards, stats, prefix):
        self.logger.log_stat(prefix + "_return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "_return_std", np.std(returns), self.t_env)
        returns.clear()

        self.logger.log_stat(prefix + "_reward_mean", np.mean(rewards), self.t_env)
        self.logger.log_stat(prefix + "_reward_std", np.std(rewards), self.t_env)
        rewards.clear()

        for k, v in stats.items():
            if k not in ["n_episodes", "individual_rewards"]:
                self.logger.log_stat(
                    prefix + "_" + k + "_mean", v / stats["n_episodes"], self.t_env
                )

        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            MAX_RETRIES = 10
            RETRY_DELAY = 0.5
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    reward, terminated, env_info = env.step(actions)
                    break
                except Exception as e:
                    print(f"Error! Retrying...")
                    time.sleep(RETRY_DELAY)
                    retries += 1
            else:
                print(
                    f"Failed to communicate with environment after {MAX_RETRIES} attempts."
                )
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
