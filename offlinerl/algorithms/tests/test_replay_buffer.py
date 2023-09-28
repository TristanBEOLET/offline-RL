import numpy as np
import pytest
import torch

from offlinerl.algorithms.replay_buffer import ReplayBuffer


@pytest.fixture
def setup_buffer():
    state_dim = 4
    action_dim = 2
    max_size = 100
    batch_size = 32

    buffer = ReplayBuffer(state_dim, action_dim, max_size)
    return state_dim, action_dim, max_size, batch_size, buffer

def test_add_and_sample(setup_buffer):
    state_dim, action_dim, max_size, batch_size, buffer = setup_buffer

    state = np.ones(state_dim)
    action = np.ones(action_dim)
    next_state = np.zeros(state_dim)
    reward = 0.5
    done = False
    buffer.add(state, action, next_state, reward, done)

    states, actions, next_states, rewards, not_dones = buffer.sample(batch_size)

    assert states.shape == torch.Size([batch_size, state_dim])
    assert actions.shape == torch.Size([batch_size, action_dim])
    assert next_states.shape == torch.Size([batch_size, state_dim])
    assert rewards.shape == torch.Size([batch_size, 1])
    assert not_dones.shape == torch.Size([batch_size, 1])

def test_normalize_states(setup_buffer):
    state_dim, action_dim, max_size, batch_size, buffer = setup_buffer

    state = np.random.rand(max_size, state_dim)
    action = np.random.rand(max_size, action_dim)
    next_state = np.random.rand(max_size, state_dim)
    reward = np.random.rand(max_size, 1)
    done = np.random.rand(max_size, 1) < 0.5
    for i in range(max_size):
        buffer.add(state[i], action[i], next_state[i], reward[i], done[i])

    mean, std = buffer.normalize_states()

    assert mean.shape == (1, state_dim)
    assert std.shape == (1, state_dim)

    normalized_state = (state - mean) / std
    normalized_next_state = (next_state - mean) / std
    np.testing.assert_allclose(buffer.state[: buffer.size], normalized_state)
    np.testing.assert_allclose(buffer.next_state[: buffer.size], normalized_next_state)


def test_save_and_load(setup_buffer):
    state_dim, action_dim, max_size, batch_size, buffer = setup_buffer

    state = np.random.rand(max_size, state_dim)
    action = np.random.rand(max_size, action_dim)
    next_state = np.random.rand(max_size, state_dim)
    reward = np.random.rand(max_size, 1)
    done = np.random.rand(max_size, 1) < 0.5
    for i in range(max_size):
        buffer.add(state[i], action[i], next_state[i], reward[i], done[i])

    filepath = "replay_buffer.pkl"
    buffer.save(filepath)

    loaded_buffer = ReplayBuffer.load(filepath)

    assert loaded_buffer.max_size == buffer.max_size
    assert loaded_buffer.size == buffer.size
    np.testing.assert_allclose(loaded_buffer.state[: loaded_buffer.size], buffer.state[: buffer.size])
    np.testing.assert_allclose(loaded_buffer.action[: loaded_buffer.size], buffer.action[: buffer.size])
    np.testing.assert_allclose(loaded_buffer.next_state[: loaded_buffer.size], buffer.next_state[: buffer.size])
    np.testing.assert_allclose(loaded_buffer.reward[: loaded_buffer.size], buffer.reward[: buffer.size])
    np.testing.assert_allclose(loaded_buffer.not_done[: loaded_buffer.size], buffer.not_done[: buffer.size])

def test_replace(setup_buffer):
    state_dim, action_dim, max_size, batch_size, buffer = setup_buffer

    indices = np.array([0, 1, 2])
    new_states = np.random.randn(len(indices), state_dim)
    new_actions = np.random.randn(len(indices), action_dim)
    new_next_states = np.random.randn(len(indices), state_dim)
    new_rewards = np.random.randn(len(indices), 1)
    new_not_dones = np.random.randn(len(indices), 1)

    buffer._replace(indices, new_states, new_actions, new_next_states, new_rewards, new_not_dones)

    assert np.allclose(buffer.state[:3], new_states)
    assert np.allclose(buffer.action[:3], new_actions)
    assert np.allclose(buffer.next_state[:3], new_next_states)
    assert np.allclose(buffer.reward[:3], new_rewards)
    assert np.allclose(buffer.not_done[:3], new_not_dones)

def test_replace_end(setup_buffer):
    state_dim, action_dim, max_size, batch_size, buffer = setup_buffer

    replace_len = 5
    new_states = np.random.randn(replace_len, state_dim)
    new_actions = np.random.randn(replace_len, action_dim)
    new_next_states = np.random.randn(replace_len, state_dim)
    new_rewards = np.random.randn(replace_len, 1)
    new_not_dones = np.random.randn(replace_len, 1)

    buffer.replace_end(new_states, new_actions, new_next_states, new_rewards, new_not_dones)

    assert np.allclose(buffer.state[-replace_len:], new_states)
    assert np.allclose(buffer.action[-replace_len:], new_actions)
    assert np.allclose(buffer.next_state[-replace_len:], new_next_states)
    assert np.allclose(buffer.reward[-replace_len:], new_rewards)
    assert np.allclose(buffer.not_done[-replace_len:], new_not_dones)

def test_replace_end_with_buffer(setup_buffer):
    state_dim, action_dim, max_size, batch_size, buffer = setup_buffer

    num_samples = 10
    other_buffer = ReplayBuffer(state_dim, action_dim, num_samples)
    states = np.random.randn(num_samples, state_dim)
    actions = np.random.randn(num_samples, action_dim)
    next_states = np.random.randn(num_samples, state_dim)
    rewards = np.random.randn(num_samples, 1)
    not_dones = np.random.randint(0, 2, size=(num_samples, 1))

    for i in range(num_samples):
        other_buffer.add(states[i], actions[i], next_states[i], rewards[i], 1 - not_dones[i])

    buffer.replace_end_with_buffer(other_buffer)

    assert np.array_equal(buffer.state[-num_samples:], states)
    assert np.array_equal(buffer.action[-num_samples:], actions)
    assert np.array_equal(buffer.next_state[-num_samples:], next_states)
    assert np.array_equal(buffer.reward[-num_samples:], rewards)
    assert np.array_equal(buffer.not_done[-num_samples:], not_dones)

def test_add_batch(setup_buffer):
    state_dim, action_dim, max_size, batch_size, buffer = setup_buffer

    batch_size = 5
    states = np.random.rand(batch_size, state_dim)
    actions = np.random.rand(batch_size, action_dim)
    next_states = np.random.rand(batch_size, state_dim)
    rewards = np.random.rand(batch_size, 1)
    dones = np.random.rand(batch_size, 1) < 0.5

    buffer.add_batch(states, actions, next_states, rewards, dones)

    assert buffer.size == batch_size
    np.testing.assert_allclose(buffer.state[:batch_size], states)
    np.testing.assert_allclose(buffer.action[:batch_size], actions)
    np.testing.assert_allclose(buffer.next_state[:batch_size], next_states)
    np.testing.assert_allclose(buffer.reward[:batch_size], rewards)
    np.testing.assert_allclose(buffer.not_done[:batch_size], 1.0 - dones)

    new_states = np.random.rand(max_size, state_dim)
    new_actions = np.random.rand(max_size, action_dim)
    new_next_states = np.random.rand(max_size, state_dim)
    new_rewards = np.random.rand(max_size, 1)
    new_dones = np.random.rand(max_size, 1) < 0.5

    buffer.add_batch(new_states, new_actions, new_next_states, new_rewards, new_dones)

    assert buffer.size == max_size
    np.testing.assert_allclose(buffer.state[:batch_size], new_states[-batch_size:])
    np.testing.assert_allclose(buffer.action[:batch_size], new_actions[-batch_size:])
    np.testing.assert_allclose(buffer.next_state[:batch_size], new_next_states[-batch_size:])
    np.testing.assert_allclose(buffer.reward[:batch_size], new_rewards[-batch_size:])
    np.testing.assert_allclose(buffer.not_done[:batch_size], 1.0 - new_dones[-batch_size:])
    np.testing.assert_allclose(buffer.state[batch_size:], new_states[:-batch_size])
    np.testing.assert_allclose(buffer.action[batch_size:], new_actions[:-batch_size])
    np.testing.assert_allclose(buffer.next_state[batch_size:], new_next_states[:-batch_size])
    np.testing.assert_allclose(buffer.reward[batch_size:], new_rewards[:-batch_size])
    np.testing.assert_allclose(buffer.not_done[batch_size:], 1.0 - new_dones[:-batch_size])
