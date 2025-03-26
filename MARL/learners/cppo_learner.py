import torch
from torch.optim import Adam

from MARL.components.action_selectors import categorical_entropy
from MARL.components.episode_buffer import EpisodeBatch
from MARL.modules.critics import REGISTRY as critic_registry
from MARL.utils.rl_utils import build_gae_targets, build_gae_targets_with_T
from MARL.utils.value_norm import ValueNorm
from MARL.modules.cost_estimators.cost_estimator import CostEstimator
import wandb


class CPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.config["learner_log_interval"] - 1

        # a trick to reuse mac
        # dummy_args = copy.deepcopy(args)
        # dummy_args.n_actions = 1
        # self.critic = NMAC(scheme, None, dummy_args)
        self.device = "cuda" if args.use_cuda else "cpu"
        self.critic = critic_registry[args.config["critic_type"]](scheme, args)
        self.cost_critic = critic_registry[args.config["critic_type"]](scheme, args).to(
            self.device
        )
        self.params = (
            list(self.mac.parameters())
            + list(self.critic.parameters())
            + list(self.cost_critic.parameters())
        )

        self.cost_estimator = CostEstimator(
            scheme["state"]["vshape"], self.n_agents, args
        ).to(self.device)

        self.optimiser = Adam(params=self.params, lr=args.config["lr"])
        self.last_lr = args.config["lr"]

        self.cost_estimator = CostEstimator(
            scheme["state"]["vshape"], self.n_agents, args
        )
        self.cost_estimator_optimiser = Adam(
            params=self.cost_estimator.parameters(),
            lr=args.config.get("cost_lr", 0.001),
        )

        self.lambda_param = torch.nn.Parameter(
            torch.tensor(args.config["lambda_init"]), requires_grad=True
        )
        self.lambda_optimiser = Adam([self.lambda_param], lr=args.config["lambda_lr"])

        self.use_value_norm = self.args.config.get("use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=self.args.device)
            self.cost_value_norm = ValueNorm(1, device=self.args.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        if self.args.config["use_individual_rewards"]:
            rewards = batch["individual_rewards"][:, :-1].to(batch.device)
            costs = batch["individual_costs"][:, :-1].to(batch.device)
        else:
            rewards = (
                batch["reward"][:, :-1]
                .to(batch.device)
                .unsqueeze(2)
                .repeat(1, 1, self.n_agents, 1)
            )
            costs = (
                batch["cost"][:, :-1]
                .to(batch.device)
                .unsqueeze(2)
                .repeat(1, 1, self.n_agents, 1)
            )
            if self.args.config["use_mean_team_reward"]:
                rewards /= self.n_agents
                costs /= self.n_agents

        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        old_probs = batch["probs"][:, :-1]
        old_probs[avail_actions == 0] = 1e-10
        old_logprob = torch.log(torch.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        # targets and advantages
        with torch.no_grad():
            if "rnn" in self.args.config["critic_type"]:
                old_values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    agent_outs = self.critic.forward(batch, t=t)
                    old_values.append(agent_outs)
                old_values = torch.stack(old_values, dim=1)
            else:
                old_values = self.critic(batch)

            if self.use_value_norm:
                value_shape = old_values.shape
                values = self.value_norm.denormalize(old_values.view(-1)).view(
                    value_shape
                )
            else:
                values = old_values

            advantages, targets = build_gae_targets(
                rewards * 100,  # .unsqueeze(2).repeat(1, 1, self.n_agents, 1),
                mask_agent,
                values,
                self.args.config["gamma"],
                self.args.config["gae_lambda"],
            )

        normed_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        with torch.no_grad():
            if "rnn" in self.args.config["critic_type"]:
                cost_old_values = []
                self.cost_critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    cost_agent_outs = self.cost_critic.forward(batch, t=t)
                    cost_old_values.append(cost_agent_outs)
                cost_old_values = torch.stack(cost_old_values, dim=1)
            else:
                cost_old_values = self.cost_critic(batch)

            if self.use_value_norm:
                cost_value_shape = cost_old_values.shape
                cost_values = self.cost_value_norm.denormalize(
                    cost_old_values.view(-1)
                ).view(cost_value_shape)
            else:
                cost_values = cost_old_values

            cost_advantages, cost_targets = build_gae_targets(
                costs * 100,  # .unsqueeze(2).repeat(1, 1, self.n_agents, 1),
                mask_agent,
                cost_values,
                self.args.config["gamma"],
                self.args.config["gae_lambda"],
            )

        normed_cost_advantages = (cost_advantages - cost_advantages.mean()) / (
            cost_advantages.std() + 1e-6
        )

        # PPO Loss
        for _ in range(self.args.config["mini_epochs"]):
            # Critic
            if "rnn" in self.args.config["critic_type"]:
                values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length - 1):
                    agent_outs = self.critic.forward(batch, t=t)
                    values.append(agent_outs)
                values = torch.stack(values, dim=1)
            else:
                values = self.critic(batch)[:, :-1]

            td_error = (values - targets.detach()) ** 2
            masked_td_error = td_error * mask_agent
            critic_loss = 0.5 * masked_td_error.sum() / mask_agent.sum()

            if "rnn" in self.args.config["critic_type"]:
                cost_values = []
                self.cost_critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length - 1):
                    cost_agent_outs = self.cost_critic.forward(batch, t=t)
                    cost_values.append(cost_agent_outs)
                cost_values = torch.stack(cost_values, dim=1)
            else:
                cost_values = self.cost_critic(batch)[:, :-1]

            cost_td_error = (cost_values - cost_targets.detach()) ** 2
            cost_masked_td_error = cost_td_error * mask_agent
            cost_critic_loss = 0.5 * cost_masked_td_error.sum() / mask_agent.sum()

            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t, t_env)
                pi.append(agent_outs)
            pi = torch.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = torch.gather(pi, dim=3, index=actions)
            log_pi_taken = torch.log(pi_taken)

            ratios = torch.exp(log_pi_taken - old_logprob)
            surr1 = ratios * normed_advantages
            surr2 = (
                torch.clamp(
                    ratios,
                    1 - self.args.config["eps_clip"],
                    1 + self.args.config["eps_clip"],
                )
                * normed_advantages
            )
            actor_loss = (
                -(torch.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()
            )

            cost_surr1 = ratios * normed_cost_advantages
            cost_surr2 = (
                torch.clamp(
                    ratios,
                    1 - self.args.config["eps_clip"],
                    1 + self.args.config["eps_clip"],
                )
                * normed_cost_advantages
            )
            cost_actor_loss = (
                -(torch.min(cost_surr1, cost_surr2) * mask_agent).sum()
                / mask_agent.sum()
            )

            # entropy
            entropy_loss = categorical_entropy(pi).mean(
                -1, keepdim=True
            )  # mean over agents
            entropy_loss[mask == 0] = 0  # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()

            reward_loss = (
                actor_loss
                + self.args.config["critic_coef"] * critic_loss
                - self.args.config["entropy_coef"] * entropy_loss
            )

            cost_loss = (
                cost_actor_loss
                + self.args.config["critic_coef"] * cost_critic_loss
                - self.args.config["entropy_coef"] * entropy_loss
            )
            loss = reward_loss - self.lambda_param * cost_loss
            # Optimise agents
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.params, self.args.config["grad_norm_clip"]
            )
            self.optimiser.step()
            cost_estimate, estimate_loss = self.train_cost_estimator(batch)

            if self.args.ablation == 'fixed_lambda':
                pass
            elif self.args.ablation == "advantage_lambda_update":
                with torch.no_grad():
                    mean_cost_advantages = cost_advantages.mean(dim=(0, 1))
                    constraint_violation = mean_cost_advantages - self.args.config["cost_limit"]
                    delta_lambda = -constraint_violation.mean()

                self.lambda_optimiser.zero_grad()
                self.lambda_param.grad = delta_lambda
                self.lambda_optimiser.step()

                with torch.no_grad():
                    self.lambda_param.clamp_(min=0.0)
            else:
                self._update_lambda(cost_estimate.detach())

            if _ == self.args.config["mini_epochs"] - 1 and self.args.enable_wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "values": values.mean().item(),
                        "targets": targets.mean().item(),
                        "reward": rewards.mean(),
                        "estimate_loss": estimate_loss.item(),
                        "lambda": self.lambda_param.item(),
                        "cost_critic_loss": cost_critic_loss.item(),
                        "cost_values": cost_values.mean().item(),
                        "cost_targets": targets.mean().item(),
                        "cost_estimate": cost_estimate.mean().item(),
                        "costs": costs.mean().item(),
                    }
                )

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

        with torch.no_grad():
            self.lambda_param.clamp_(min=0.0)
        return lambda_loss

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path, postfix=""):
        self.mac.save_models(path, postfix)
        torch.save(
            self.optimiser.state_dict(), "{}/agent_opt".format(path) + postfix + ".th"
        )

    def load_models(self, path, postfix=""):
        self.mac.load_models(path, postfix)
        # Not quite right but I don't want to save target networks
        self.optimiser.load_state_dict(
            torch.load(
                "{}/agent_opt".format(path) + postfix + ".th",
                map_location=lambda storage, loc: storage,
            )
        )
