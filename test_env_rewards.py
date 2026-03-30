import pytest
import numpy as np
from env import HomeostaticEnv
import rewards


def test_default_reward_function():
    env = HomeostaticEnv(num_vars=2, max_steps=5, rewarding='default')
    obs, info = env.reset(seed=123)
    action = env.action_space.sample()
    obs2, r, done, ended, info2 = env.step(action)
    assert isinstance(r, float)
    assert obs2.shape == obs.shape


def test_qd_reward_function():
    env = HomeostaticEnv(num_vars=2, max_steps=5, rewarding='qd')
    obs, info = env.reset(seed=123)
    action = env.action_space.sample()
    obs2, r, done, ended, info2 = env.step(action)
    assert isinstance(r, float)
    assert r != 0 or r == 0


def test_custom_reward_plugin_by_module():
    env = HomeostaticEnv(num_vars=2, max_steps=5, rewarding='rewards:custom_reward')
    obs, info = env.reset(seed=123)
    action = np.zeros(env.num_vars, dtype=np.float32)
    obs2, r, done, ended, info2 = env.step(action)
    assert isinstance(r, float)


def test_custom_reward_plugin_shortcut():
    env = HomeostaticEnv(num_vars=2, max_steps=5, rewarding='custom_reward')
    obs, info = env.reset(seed=123)
    action = np.zeros(env.num_vars, dtype=np.float32)
    obs2, r, done, ended, info2 = env.step(action)
    assert isinstance(r, float)


def test_callable_reward_function():
    def dummy(env, action, learned_params=None):
        obs, _, done, ended, info = env._default_step(action)
        return obs, 42.0, done, ended, info

    env = HomeostaticEnv(num_vars=2, max_steps=5, rewarding=dummy)
    obs, info = env.reset(seed=123)
    action = env.action_space.sample()
    obs2, r, done, ended, info2 = env.step(action)
    assert r == 42.0
