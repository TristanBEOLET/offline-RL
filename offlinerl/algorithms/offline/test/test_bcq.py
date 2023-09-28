import os

import numpy as np
import pytest

from offlinerl.algorithms.offline.bcq import BCQ
from offlinerl.algorithms.offline.cql import CQL
from offlinerl.algorithms.offline.td3bc import TD3_BC
from offlinerl.algorithms.online.td3 import TD3
from offlinerl.algorithms.replay_buffer import ReplayBuffer


# Create a fixture that returns an instance of the algorithm
@pytest.fixture(params=[BCQ, TD3_BC, CQL, TD3])
def agent_instance(request):
    return request.param(state_dim=3, action_dim=2, max_action=1)


def test_select_action(agent_instance):
    state = np.array([1, 2, 3])
    action = agent_instance.select_action(state)
    assert isinstance(action, np.ndarray)

def test_train_step(agent_instance):
    replay_buffer = ReplayBuffer(state_dim=3, action_dim=2, max_size=10)
    replay_buffer.add_batch(states=np.zeros((10,3)), actions=np.zeros((10,2)), next_states=np.zeros((10,3)), rewards=np.zeros((10,1)), dones=np.zeros((10,1)))
    agent_instance.train_step(replay_buffer)

def test_save_load(agent_instance):
    filename = "test_save_load.pkl"
    
    agent_instance.save(filename)
    assert os.path.exists(filename)
    
    agent_instance.load(filename, state_dim=3, action_dim=2, max_action=1, **{})
    os.remove(filename)
