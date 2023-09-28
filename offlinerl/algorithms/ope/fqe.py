from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from beartype.typing import Tuple
from tqdm import tqdm

from offlinerl.algorithms.offline.td3bc import TD3_BC
from offlinerl.algorithms.replay_buffer import ReplayBuffer
from offlinerl.neural_networks.critic import TwinCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FQE:
    """
    Fitted Q evaluation
    https://arxiv.org/pdf/1903.08738.pdf
    """

    def __init__(
        self,
        agent: TD3_BC,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        discount: float,
        lr: float = 3e-4,
        target_update_interval: int = 100,
        hidden_dims: Tuple[int] = (256, 256),
    ) -> None:
        """
        :param agent: the agent to evaluate
        """
        self.agent = agent

        self.q_network = TwinCritic(state_dim, action_dim, hidden_dims=list(hidden_dims)).to(device)
        self.target_q_network = copy.deepcopy(self.q_network)

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.total_it = 0

        self.batch_size = batch_size
        self.discount = discount
        self.target_update_interval = target_update_interval

    def train_step(self, replay_buffer: ReplayBuffer) -> None:
        """
        One training step of the FQE
        :param replay_buffer: the replay buffer from which to sample transitions
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

        # Computing the target:
        with torch.no_grad():
            next_action = self.agent.actor(next_state)
            target_q = reward + not_done * self.discount * self.target_q_network.Q1(
                next_state, next_action
            )

        # Loss from Bellman:
        pred_q = self.q_network.Q1(state, action)
        loss = F.mse_loss(target_q, pred_q)

        # Optimisation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network periodic update
        if self.total_it % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def full_training(self, replay_buffer: ReplayBuffer, n_steps: int) -> None:
        """
        A full training of the FQE
        :param replay_buffer: the replay_buffer
        :param n_steps: how many gradient steps to do overall
        NB: n_steps > 50k for sensible results, preferably 100k
        """
        # OneCycleLR is fairly robust.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, total_steps=n_steps
        )

        test_states, _, _, _, _ = replay_buffer.sample(100)

        progress_bar = tqdm(total=n_steps, position=0, leave=True)

        for o in range(n_steps):
            self.train_step(replay_buffer)
            scheduler.step()
            progress_bar.update(1)
            if o % (n_steps // 100) == 0:
                value = torch.mean(self.evaluate_q_values(test_states)).item()
                progress_bar.set_description(f"Mean Q Values: {value:.2f}")

    def evaluate_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of states, returns the estimated values of the states under the agent policy
        :param states: the states on which to evaluate
        """
        actions = self.agent.actor(states)
        q_values = self.q_network.Q1(states, actions)
        return q_values

    def save(self, filename: str) -> None:
        """
        Save FQE model
        :param filename: filename for the .pt file containing all the networks
        """
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "q_network_optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    @staticmethod
    def load(
        filename: str,
        agent: TD3_BC,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        discount: float,
        lr: float = 3e-4,
        target_update_interval: int = 100,
        hidden_dims: Tuple[int] = (256, 256),
    ) -> FQE:
        """
        Loads the saved FQE
        :param filename: filename of the .pt file
        :param agent, state_dim, action_dim, discount, batch_size, lr, target_update_interval:
        see constructor
        """
        fqe = FQE(
            agent=agent,
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            batch_size=batch_size,
            discount=discount,
            target_update_interval=target_update_interval,
            hidden_dims=hidden_dims,
        )

        checkpoint = torch.load(filename)
        fqe.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        fqe.target_q_network.load_state_dict(checkpoint["q_network_state_dict"])
        fqe.optimizer.load_state_dict(checkpoint["q_network_optimizer_state_dict"])

        return fqe


# # Example of usage:
# if __name__ == "__main__":
#     import omegaconf
#     from algo_design_validation.algo_design_validation.dlrl.control.rl.offline_model_free import (
#         utils,
#     )
#
#     cfg = omegaconf.OmegaConf.load(
#         "/home/maxime/Documents/offline_paper_data/debug_fqe/baseline_conf.yaml"
#     )
#
#     # Just changing the patients so that it's faster for this demo
#     cfg.diabeloop_real_data_params.in_max_patients = 2
#
#     env = utils.build_env(cfg)
#
#     params = omegaconf.OmegaConf.to_container(cfg.offline_agent.params)
#
#     model = TD3_BC.load(
#         "/home/maxime/Documents/offline_paper_data/debug_fqe/baseline.pt",
#         agent_params=params,
#         env=env,
#     )
#
#     cfg.environment.reward_fun = "binary_reward"
#
#     offline_dataset = utils.load_offline_dataset(cfg)
#
#     fqe_ = FQE(
#         agent=model,
#         state_dim=env.reset()[0].shape[0],
#         action_dim=model.action_dim,
#         batch_size=64,
#         discount=0.99,
#         target_update_interval=100,
#     )
#
#     fqe_.full_training(offline_dataset, n_steps=100000)
