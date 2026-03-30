"""Módulo de recompensas plugin-and-play para HomeostaticEnv.

O ambiente aceita o parâmetro "rewarding" como:
- callable(env, action, learned_params=None)
- 'default', 'qd', 'operational_regimes', 'OR', 'OR_full'
- 'rewards:default_reward' (módulo rewards.py local), ou outro módulo:func
- 'func_name' como atalho para 'rewards:func_name'

Exemplos:
    env = HomeostaticEnv(rewarding='default')
    env = HomeostaticEnv(rewarding='qd')
    env = HomeostaticEnv(rewarding='rewards:custom_reward')
    env = HomeostaticEnv(rewarding='minha_lib:custom_reward')

Cada função de recompensa deve retornar exatamente a tupla (obs, reward, done, ended, info),
como espera a API Gymnasium do método step().
"""


def default_reward(env, action, learned_params=None):
    """Recompensa padrão: negativo da dist[ância quadrática à target."""
    return env._default_step(action)


def qd_reward(env, action, learned_params=None):
    """Recompensa QD: melhoria em relação ao passo anterior."""
    return env._qd_step(action)


def operationalregimes_reward(env, action, learned_params=None):
    """Recompensa operacional de regimes (OR)."""
    return env._operationalregimes_step(action, learned_params=learned_params)


def custom_reward(env, action, learned_params=None):
    """Exemplo de plugin de recompensa customizado.

    return:
        obs, reward, done, ended, info
    """
    obs, _, done, ended, info = env._default_step(action)
    penalty = sum(abs(x) for x in action) * 0.1
    reward = -sum((s - t) ** 2 for s, t in zip(env.state, env.target_values)) - penalty
    return obs, reward * env.reward_scale, done, ended, info
