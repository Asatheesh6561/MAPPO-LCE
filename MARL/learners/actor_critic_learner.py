# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy

import torch as th
from torch.optim import Adam

from MARL.components.episode_buffer import EpisodeBatch
from MARL.components.standardize_stream import RunningMeanStd
from MARL.modules.critics import REGISTRY as critic_registry
from MARL.modules.cost_estimators.cost_estimator import CostEstimator
import wandb


class ActorCriticLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.device = "cuda" if args.use_cuda else "cpu"
        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.config["lr"])

        self.critic = critic_registry[args.config["critic_type"]](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.cost_critic = critic_registry[args.config["critic_type"]](scheme, args).to(
            self.device
        )
        self.target_cost_critic = copy.deepcopy(self.cost_critic).to(self.device)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.config["lr"])
        self.cost_critic_params = list(self.cost_critic.parameters())
        self.cost_critic_optimiser = Adam(
            params=self.cost_critic_params, lr=args.config["lr"]
        )

        self.cost_estimator = CostEstimator(
            scheme["state"]["vshape"], self.n_agents, args
        ).to(self.device)
        self.cost_estimator_optimiser = Adam(
            params=self.cost_estimator.parameters(),
            lr=args.config.get("cost_lr", 0.001),
        )

        self.lambda_param = th.nn.Parameter(
            th.tensor(args.config["lambda_init"]), requires_grad=True
        )
        self.lambda_optimiser = Adam(
            [self.lambda_param], lr=args.config.get("lambda_lr", 0.001)
        )

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.config["learner_log_interval"] - 1

        if self.args.config["standardise_returns"]:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.config["standardise_rewards"]:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)
        if self.args.config["standardise_costs"]:
            self.cost_ms = RunningMeanStd(shape=(1,), device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        costs = batch["costs"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.config["standardise_rewards"]:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error(
                "Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env)
            )
            return

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t, t_env)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        pi = mac_out
        advantages, critic_train_stats = self.train_critic_sequential(
            self.critic,
            self.target_critic,
            batch,
            rewards,
            critic_mask,
        )
        cost_advantages, cost_critic_train_stats = self.train_cost_critic_sequential(
            self.cost_critic,
            self.target_cost_critic,
            batch,
            costs,
            critic_mask,
        )
        actions = actions[:, :-1]
        advantages = advantages.detach()
        cost_advantages = cost_advantages.detach()

        pg_loss = self._calculate_policy_loss(
            pi, actions, mask, advantages, cost_advantages
        )

        self.agent_optimiser.zero_grad()
        pg_loss.backward(retain_graph=True)  # Retain graph for following calculations
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.agent_params, self.args.config["grad_norm_clip"]
        )
        self.agent_optimiser.step()

        cost_estimate, cost_estimate_loss = self.train_cost_estimator(batch)

        lambda_loss = self._update_lambda(
            cost_estimate.detach()
        )  # Detach cost_estimate for lambda update
        self._update_targets()
        self.critic_training_steps += 1

        ts_logged = len(critic_train_stats["critic_loss"])
        if t_env - self.log_stats_t >= self.args.config["learner_log_interval"]:

            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )
                self.logger.log_stat(
                    "cost_" + key,
                    sum(cost_critic_train_stats["cost_" + key]) / ts_logged,
                    t_env,
                )

            self.logger.log_stat("lambda", self.lambda_param.item(), t_env)
            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("lambda_loss", lambda_loss.item(), t_env)
            self.logger.log_stat("cost_estimate_loss", cost_estimate_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env

        if self.args.enable_wandb:
            wandb.log(
                {
                    "critic_loss": sum(critic_train_stats["critic_loss"]) / ts_logged,
                    "critic_grad_norm": sum(critic_train_stats["critic_grad_norm"])
                    / ts_logged,
                    "td_error_abs": sum(critic_train_stats["td_error_abs"]) / ts_logged,
                    "q_taken_mean": sum(critic_train_stats["q_taken_mean"]) / ts_logged,
                    "target_mean": sum(critic_train_stats["target_mean"]) / ts_logged,
                    "cost_critic_loss": sum(cost_critic_train_stats["cost_critic_loss"])
                    / ts_logged,
                    "cost_critic_grad_norm": sum(
                        cost_critic_train_stats["cost_critic_grad_norm"]
                    )
                    / ts_logged,
                    "cost_td_error_abs": sum(
                        cost_critic_train_stats["cost_td_error_abs"]
                    )
                    / ts_logged,
                    "cost_q_taken_mean": sum(
                        cost_critic_train_stats["cost_q_taken_mean"]
                    )
                    / ts_logged,
                    "cost_target_mean": sum(cost_critic_train_stats["cost_target_mean"])
                    / ts_logged,
                    "lambda": self.lambda_param.item(),
                    "advantage_mean": (advantages * mask).sum().item()
                    / mask.sum().item(),
                    "cost_advantage_mean": (cost_advantages * mask).sum().item()
                    / mask.sum().item(),
                    "pg_loss": pg_loss.item(),
                    "lambda_loss": lambda_loss.item(),
                    "cost_estimate_loss": cost_estimate_loss.item(),
                    "agent_grad_norm": grad_norm.item(),
                    "pi_max": (pi.max(dim=-1)[0] * mask).sum().item()
                    / mask.sum().item(),
                },
                step=t_env,
            )

    def _calculate_policy_loss(self, pi, actions, mask, advantages, cost_advantages):
        pi[mask == 0] = 1.0
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)

        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        pg_loss = (
            -(
                advantages * log_pi_taken
                - self.lambda_param * cost_advantages * log_pi_taken
                + self.args.config["entropy_coef"] * entropy
            )
            * mask
        ).sum() / mask.sum()
        return pg_loss

    def train_cost_estimator(self, batch):
        state = batch["state"][:, :-1].to(self.device)
        actions = batch["actions"][:, :-1].squeeze(3).to(self.device)
        cost_estimate = self.cost_estimator(state, actions)
        estimate_loss = (batch["costs"][:, :-1] - cost_estimate).pow(2).mean()
        self.cost_estimator_optimiser.zero_grad()
        estimate_loss.backward(retain_graph=True)  # Retain graph for subsequent updates
        self.cost_estimator_optimiser.step()
        return cost_estimate, estimate_loss

    def _update_lambda(self, cost_estimate):
        cost_estimate = cost_estimate.detach()
        constraint_violation = cost_estimate - self.args.config["cost_limit"]
        lambda_loss = -self.lambda_param * constraint_violation.mean()

        self.lambda_optimiser.zero_grad()
        lambda_loss.backward()
        self.lambda_optimiser.step()

        with th.no_grad():
            self.lambda_param.clamp_(min=0.0)
        return lambda_loss

    def _update_targets(self):
        # Hard or soft update of target networks based on specified interval
        if (
            self.args.config["target_update_interval_or_tau"] > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.config["target_update_interval_or_tau"]
            >= 1.0
        ):
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_cost_critic.load_state_dict(self.cost_critic.state_dict())
            self.last_target_update_step = self.critic_training_steps
        elif self.args.config["target_update_interval_or_tau"] <= 1.0:
            tau = self.args.config["target_update_interval_or_tau"]
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
            for target_param, param in zip(
                self.target_cost_critic.parameters(), self.cost_critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            if "rnn" in self.args.config["critic_type"]:
                old_values = []
                target_critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    agent_outs = target_critic.forward(batch, t=t)
                    old_values.append(agent_outs)
                target_vals = th.stack(old_values, dim=1)
            else:
                target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.config["standardise_returns"]:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.config["q_nstep"]
        )

        if self.args.config["standardise_returns"]:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        if "rnn" in self.args.config["critic_type"]:
            values = []
            critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = critic.forward(batch, t=t)
                values.append(agent_outs)
            values = th.stack(values, dim=1)
        else:
            values = critic(batch)[:, :-1]
        values = values.squeeze(3)
        td_error = target_returns.detach() - values
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.config["grad_norm_clip"]
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((values * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )
        return masked_td_error, running_log

    def train_cost_critic_sequential(
        self, cost_critic, target_cost_critic, batch, costs, mask
    ):
        # Optimise critic
        with th.no_grad():
            if "rnn" in self.args.config["critic_type"]:
                old_values = []
                target_cost_critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    agent_outs = target_cost_critic.forward(batch, t=t)
                    old_values.append(agent_outs)
                target_vals = th.stack(old_values, dim=1)
            else:
                target_vals = target_cost_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.config["standardise_returns"]:
            target_vals = (
                target_vals * th.sqrt(self.cost_ret_ms.var) + self.cost_ret_ms.mean
            )

        target_costs = self.nstep_returns(
            costs, mask, target_vals, self.args.config["q_nstep"]
        )

        if self.args.config["standardise_costs"]:
            self.cost_ms.update(target_costs)
            target_costs = (target_costs - self.cost_ms.mean) / th.sqrt(
                self.cost_ms.var
            )

        running_log = {
            "cost_critic_loss": [],
            "cost_critic_grad_norm": [],
            "cost_td_error_abs": [],
            "cost_target_mean": [],
            "cost_q_taken_mean": [],
        }

        if "rnn" in self.args.config["critic_type"]:
            values = []
            cost_critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = cost_critic.forward(batch, t=t)
                values.append(agent_outs)
            values = th.stack(values, dim=1)
        else:
            values = cost_critic(batch)[:, :-1]
        values = values.squeeze(3)
        td_error = target_costs.detach() - values
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.cost_critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.cost_critic_params, self.args.config["grad_norm_clip"]
        )
        self.cost_critic_optimiser.step()

        running_log["cost_critic_loss"].append(loss.item())
        running_log["cost_critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["cost_td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["cost_q_taken_mean"].append(
            (values * mask).sum().item() / mask_elems
        )
        running_log["cost_target_mean"].append(
            (target_costs * mask).sum().item() / mask_elems
        )
        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.config["gamma"] ** step * values[:, t] * mask[:, t]
                    )
                elif t == rewards.size(1) - 1 and self.args.config.get(
                    "add_value_last_step", False
                ):
                    nstep_return_t += (
                        self.args.config["gamma"] ** step * rewards[:, t] * mask[:, t]
                    )
                    nstep_return_t += (
                        self.args.config["gamma"] ** (step + 1) * values[:, t + 1]
                    )
                else:
                    nstep_return_t += (
                        self.args.config["gamma"] ** step * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
