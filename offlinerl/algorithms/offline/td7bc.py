from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn.functional as F
from beartype.typing import Dict

from offlinerl.algorithms.lap_replay_buffer import LAP
from offlinerl.algorithms.online.td7 import TD7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD7_BC(TD7):
    """
    Implementation of the TD7 algorithm with behavior cloning (BC).

    Inherits from TD7 and adds behavior cloning regularization to the actor loss.

    :param kwargs: Keyword arguments to pass to the parent TD7 class.
    """

    def __init__(self, nb_batch_cloning, **kwargs):
        """
        Initialize the TD7_BC agent.

        :param kwargs: Keyword arguments to pass to the parent TD7 class.
        """
        super().__init__(offline=True, **kwargs)
        self.rl_lmbda = kwargs["lmbda"]
        self.current_batch_nb = 0
        self.nb_batch_cloning = nb_batch_cloning

    def train_step(self, replay_buffer: LAP):
        """
        Perform a single training step using the given replay buffer.

        :param replay_buffer: Replay buffer containing experience samples.
        :return: Dictionary of computed losses.
        """
        if self.current_batch_nb < self.nb_batch_cloning:
            self.lmbda = 0.0
        else:
            self.lmbda = self.rl_lmbda
        train_dict = super().train(replay_buffer)
        self.current_batch_nb += 1
        return train_dict

    def compute_actor_loss(  # pylint: disable=W0222
        self, Q: torch.Tensor, action: torch.Tensor, actor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the actor loss with behavior cloning regularization.

        :param Q: Q-values from the critic.
        :param action: Selected actions.
        :param actor: Actor outputs.
        :return: Actor loss.
        """
        actor_loss = -Q.mean()
        actor_loss = actor_loss + self.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action)
        return actor_loss

    def train(self, replay_buffer: LAP):
        """
        Perform a training step using the given replay buffer.

        :param replay_buffer: Replay buffer containing experience samples.
        :return: Dictionary of computed losses.
        """
        return super().train(replay_buffer=replay_buffer)

    @staticmethod
    def load(filename: str, agent_params: Dict, env: gym.Env) -> TD7:
        """
        Load a saved model state from a file and create an instance of the TD7_BC class.

        :param filename: Name of the file to load the model state from.
        :param agent_params: Dictionary of agent parameters.
        :param env: Gym environment.
        :return: Instantiated TD7_BC object.
        """
        # More robust than env.observation_space.shape if the env is not well defined
        state_dim = env.reset()[0].shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        td7 = TD7_BC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            **agent_params,
        )

        checkpoint = torch.load(filename)
        td7.critic.load_state_dict(checkpoint["critic_state_dict"])
        td7.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        td7.actor.load_state_dict(checkpoint["actor_state_dict"])
        td7.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        td7.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        td7.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])

        return td7
