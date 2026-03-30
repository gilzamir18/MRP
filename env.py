from typing import Optional
import random
import gymnasium as gym
import numpy as np

class HomeostaticEnv(gym.Env):
    # Declare the render modes supported by this environment
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, num_vars: int=2, rewarding: str = "default",
                 obs_noise_std: float = 0.0, reward_scale: float = 1.0,
                 max_steps:int=1000, random_safety_zone=False,
                 decay_std: float = 0.001,
                 decay_ratio_mu: float = 0.001,
                 render_mode: Optional[str] = None, safety_zone: float = 0.5,
                 penalty_ratio: float = 3.0, use_learned_params: bool = False,
                 correction_budget: float = 1.0,
                 random_budget: bool = False, **kwargs):
        self.num_vars = num_vars
        self.decay_std = decay_std
        self.obs_noise_std = obs_noise_std

        self.state = [0.0] * num_vars  # Initialize all variables to 0.0
        self.survival_zone = [0.0] * num_vars
        self.target_values = [0.0] * num_vars
        self.decay_ratio_mu = decay_ratio_mu
        self.decay_ratio = [np.random.normal(decay_ratio_mu, decay_std) for _ in range(num_vars)]  # Default decay ratio for each variable
        
        self.safety_zone = [1.0] * num_vars
        self.prev_state = self.state.copy()
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.step_count = 0
        # safety_zone_value <= 0 means: randomise per variable on each reset
        self.safety_zone_value = safety_zone
        self.random_safety_zone = (safety_zone <= 0.0) or random_safety_zone
        # penalty_ratio (β): worsening penalty = -β·α, improvement reward = +α
        # β=1 → symmetric;  β=2 → current default (paper value)
        self.penalty_ratio = max(0.01, float(penalty_ratio))
        self.use_learned_params = use_learned_params
        # correction_budget ∈ (0, 1]: fraction of total action cost the agent can spend.
        # random_budget=True resamples uniformly in (0, 1] at each reset.
        self.correction_budget_value = float(np.clip(correction_budget, 0.0, 1.0))
        self.random_budget = random_budget
        self.current_budget = self.correction_budget_value

        # Validate and store render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"render_mode '{render_mode}' não suportado. "
            f"Modos válidos: {self.metadata['render_modes']}"
        )
        self.render_mode = render_mode

        # observation = state + survival_zone + safety_zone + [current_budget]
        self.observation_space = gym.spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(self.num_vars * 4 + 1,),
            dtype=float
        )

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_vars,),
            dtype=np.float32,
        )

        self.rewarding = rewarding

        
    def _get_obs(self):
        obs = np.asarray(self.state + self.survival_zone + self.safety_zone + self.decay_ratio + [self.current_budget], dtype=float)
        if self.obs_noise_std > 0.0:
            obs += np.random.normal(0.0, self.obs_noise_std, size=obs.shape)
        return obs

    def _get_info(self):
        return {
            "target_values": self.target_values,
            "decay_ratio": self.decay_ratio
        }


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        random.seed(seed)
        self.state = [ random.uniform(-0.9, 0.9) for _ in range(self.num_vars) ]
        self.survival_zone = [1.0] * self.num_vars
        self.target_values = [0.0] * self.num_vars
        self.decay_ratio = [ np.random.normal(self.decay_ratio_mu, self.decay_std) for _ in range(self.num_vars) ] #[0.001] * self.num_vars
        if self.random_safety_zone:
            self.safety_zone = [random.uniform(0.1, 0.9) for _ in range(self.num_vars)]
        else:
            self.safety_zone = [self.safety_zone_value for _ in range(self.num_vars)]
        self.prev_state = self.state.copy()
        self.step_count = 0
        if self.random_budget:
            self.current_budget = float(np.random.uniform(0.0, 1.0))
        else:
            self.current_budget = self.correction_budget_value
        return self._get_obs(), self._get_info()
    
    def step(self, action, learned_params: dict = None):
        if self.rewarding == "QD":
            return self._qd_step(action)
        elif self.rewarding in ("operational_regimes", "OR_full", "OR"):
            return self._operationalregimes_step(action, learned_params=learned_params)
        else:
            return self._default_step(action)

    def _resolve_or_params(self, learned_params=None):
        """Retorna (safety_zone_list, penalty_ratio_float), usando learned_params se disponível."""
        if learned_params is not None and self.use_learned_params:
            sz = learned_params.get('safety_zone', self.safety_zone)
            if hasattr(sz, '__len__'):
                sz = [float(sz[i]) for i in range(self.num_vars)]
            else:
                sz = [float(sz)] * self.num_vars
            beta = max(0.01, float(learned_params.get('penalty_ratio', self.penalty_ratio)))
        else:
            sz = self.safety_zone
            beta = self.penalty_ratio
        if self.use_learned_params:
            beta = beta * 10
        return sz, beta

    def _apply_budget(self, action):
        """Filter action vector according to the current correction budget.

        Variables are visited in index order.  Each variable whose |action| >
        NULL_THRESHOLD costs exactly |action[i]| from the budget.  Once the
        remaining budget is exhausted (cost > remaining), that variable's action
        is zeroed and no further variables are corrected.  Actions with
        |action| <= NULL_THRESHOLD are treated as null and are also zeroed
        without consuming any budget.

        Numerical treatment:
          - Subtrai o máximo antes de aplicar exp (estabilidade numérica, evita overflow).
          - Garante que sum_exp > 0 e finito antes de dividir.
          - Retorna distribuição uniforme como fallback se o vetor for degenerado.
        """
        action = np.asarray(action, dtype=np.float64)
        shifted = action - np.max(action)          # max → 0, demais ≤ 0
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        if sum_exp == 0.0 or not np.isfinite(sum_exp):
            return np.ones(len(action), dtype=np.float64) / len(action)
        return (exp_vals / sum_exp) * self.current_budget



    def _env_dynamic(self, action):
        min_significant = 0.001
        action = self._apply_budget(action)
        self.prev_state = self.state.copy()
        if self.decay_std > 0.0:
            self.decay_ratio = [dr + random.gauss(0.0, self.decay_std) for dr in self.decay_ratio]
        self.state = [(self.state[i] - self.decay_ratio[i]) if self.state[i] < -min_significant else  (self.state[i] + self.decay_ratio[i]) if self.state[i] > min_significant else self.state[i] for i in range(self.num_vars)]
        self.state = [self.state[i] + a if self.state[i] < -min_significant else (self.state[i] - a) if self.state[i] > min_significant else self.state[i] for i, a in enumerate(action)]
        self.step_count += 1
        return any(abs(self.state[i] - self.target_values[i]) > self.survival_zone[i] for i in range(self.num_vars))

    def _default_step(self, action):
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps      
        distance = sum((self.state[i] - self.target_values[i])**2 for i in range(self.num_vars))
        reward = -distance * self.reward_scale
        return self._get_obs(), reward, done, ended, self._get_info()
    
    def _qd_step(self, action):
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps
        dt = sum((self.state[i] - self.target_values[i]) ** 2 for i in range(self.num_vars))
        dt_1 = sum((self.prev_state[i] - self.target_values[i]) ** 2 for i in range(self.num_vars))
        reward = (dt_1 - dt) * self.reward_scale
        return self._get_obs(), reward, done, ended, self._get_info()
    
    def _operationalregimes_step(self, action, learned_params=None):
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps
        sz, beta = self._resolve_or_params(learned_params)
        reward = 0.0
        for i in range(self.num_vars):
            dt = abs(self.state[i] - self.target_values[i])
            dt_1 = abs(self.prev_state[i] - self.target_values[i])
            u = (self.state[i] <= sz[i])
            v = (dt_1 <= dt)
            reward += (u + (1 - u) * (1 - (1 + beta) * v))
        return self._get_obs(), reward * self.reward_scale, done, ended, self._get_info()


    def render(self):
        """Renderiza o estado atual do ambiente.

        Returns:
            None  se render_mode == "human"  (imprime no terminal)
            np.ndarray (H, W, 3) se render_mode == "rgb_array"
        """
        if self.render_mode == "human" or self.render_mode is None:
            # Modo padrão: imprime no terminal
            print(
                f"Step: {self.step_count} | "
                f"State: {[round(s, 4) for s in self.state]} | "
                f"Target: {self.target_values} | "
                f"Safety Zone: {[round(z, 4) for z in self.safety_zone]}"
            )
            return None

        elif self.render_mode == "rgb_array":
            # Gera uma imagem simples representando o estado como barras coloridas
            bar_width = 40
            bar_height = 120
            margin = 10
            W = (bar_width + margin) * self.num_vars + margin
            H = bar_height + 2 * margin
            img = np.ones((H, W, 3), dtype=np.uint8) * 30  # fundo escuro

            for i, val in enumerate(self.state):
                x0 = margin + i * (bar_width + margin)
                # Normaliza val de [-1, 1] para [0, bar_height]
                norm = (np.clip(float(val), -1.0, 1.0) + 1.0) / 2.0
                bar_h = int(norm * bar_height)
                y0 = margin + (bar_height - bar_h)
                y1 = margin + bar_height

                # Cor: verde se dentro da safety zone, vermelho se fora
                in_zone = abs(val - self.target_values[i]) <= self.safety_zone[i]
                color = (60, 200, 80) if in_zone else (220, 60, 60)
                img[y0:y1, x0:x0 + bar_width] = color

            return img