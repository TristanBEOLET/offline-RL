import numpy as np
import torch

from offlinerl.algorithms.replay_buffer import ReplayBuffer


class LAP(ReplayBuffer):
    """
    Loss Adjusted Prioritize (LAP) Replay Buffer.

    This class implements a replay buffer with Loss Adjusted Prioritize
    Original source code : https://github.com/sfujim/TD7/blob/main/buffer.py
    Paper : https://arxiv.org/pdf/2007.06049.pdf.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size=int(1e6), prioritized=False):
        """
        Initialize the LAP replay buffer.

        Parameters:
            state_dim : Dimension of the state space.
            action_dim : Dimension of the action space.
            max_size : Maximum size of the replay buffer. Defaults to int(1e6).
            prioritized : Whether to use prioritized replay. Defaults to True.
        """
        super().__init__(state_dim=state_dim, action_dim=action_dim, max_size=max_size)

        self.prioritized = prioritized
        if self.prioritized:
            self.priority = torch.zeros(max_size, device=self.device)
            self.max_priority = 1

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ):
        """
        Add a transition to the replay buffer.

        Parameters:
            state: The current state.
            action: The action taken.
            next_state: The next state.
            reward: The reward received.
            done: Whether the episode is done or not.
        """
        if self.prioritized:
            self.priority[self.ptr] = self.max_priority
        super().add(state=state, action=action, next_state=next_state, reward=reward, done=done)

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
            batch_size: The size of the batch to sample.

        Returns:
            A tuple of tensors containing the sampled batch of transitions and the
                indices of the batch.
        """
        if self.prioritized:
            csum = torch.cumsum(self.priority[: self.size], 0)
            val = torch.rand(size=(batch_size,), device=self.device) * csum[-1]
            sampled_indices = torch.searchsorted(csum, val).cpu().data.numpy()
            return (
                torch.FloatTensor(self.state[sampled_indices]).to(self.device),
                torch.FloatTensor(self.action[sampled_indices]).to(self.device),
                torch.FloatTensor(self.next_state[sampled_indices]).to(self.device),
                torch.FloatTensor(self.reward[sampled_indices]).to(self.device),
                torch.FloatTensor(self.not_done[sampled_indices]).to(self.device),
                sampled_indices,
            )
        else:
            return super().sample(batch_size=batch_size)

    def update_priority(self, indices: torch.Tensor, priority: torch.Tensor):
        """
        Update the priority of transitions in the replay buffer.

        Parameters:
            indices: Indices of the transitions to update (correspond to
                        self.ind in the sample method).
            priority: The updated priority values.
        """
        self.priority[indices] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        """
        Reset the maximum priority value in the replay buffer.
        """
        self.max_priority = float(self.priority[: self.size].max())
