

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


# Constants

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

SERVICE_IDX = {svc: i * len(METRICS) for i, svc in enumerate(SERVICES)}

AGENT_NEIGHBORS: Dict[str, List[str]] = {
    'cart':      ['catalogue', 'shipping'],  # input from catalogue, output to shipping
    'catalogue': ['ratings',   'cart'],      # feeds ratings and cart
    'payment':   ['shipping'],               # triggered after shipping confirm
    'shipping':  ['cart',      'payment'],   # input from cart, output to payment
    'ratings':   ['catalogue'],              # only triggered by catalogue
    'user':      ['cart'],                   # session -> cart flow
}

# Action space (from the agent's perspective)
# For DQN/PPO compatibility, 0,1,2 -> converted internally to -1,0,+1
ACTION_MAP = {0: -1, 1: 0, 2: 1}   # discrete -> internal
ACTION_MAP_INV = {-1: 0, 0: 1, 1: 2}

N_ACTIONS = 3        # {scale_down, keep, scale_up}
MAX_STEPS = 10       # DT trajectory reliability limit



class DigitalTwin(nn.Module):
    def __init__(self, input_dim=36, output_dim=30,
                 hidden_dims=(256, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# MARL Environment

class MarlEnvironment:
    """
    Multi-agent RL environment built on top of the Digital Twin.

    Usage:
        env = MarlEnvironment()
        obs = env.reset(initial_state)   # Dict[str, np.ndarray]
        obs, rewards, done, info = env.step(actions)  # actions: Dict[str, int]

    Parameters:
        dt_path         : Digital Twin model file (.pth)
        reward_cfg_path : Reward configuration file (.json)
        scaler_path     : MinMaxScaler file (.pkl)
        seed            : Random seed for reproducibility
    """

    def __init__(
        self,
        dt_path: str = 'digital_twin_best.pth',
        reward_cfg_path: str = 'reward_config.json',
        scaler_path: str = 'scaler.pkl',
        seed: Optional[int] = 42,
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device('cpu')  # CPU is sufficient for inference

        # Load Digital Twin
        self.dt = DigitalTwin().to(self.device)
        self.dt.load_state_dict(
            torch.load(dt_path, map_location=self.device, weights_only=False)
        )
        self.dt.eval()

        # Load reward configuration
        with open(reward_cfg_path, 'r') as f:
            self.reward_cfg = json.load(f)

        # Load scaler (can be used later for denormalization)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Observation space sizes (per service)
        # Own 5 metrics + 5 metrics for each neighbor
        self.obs_dims: Dict[str, int] = {
            svc: (1 + len(AGENT_NEIGHBORS[svc])) * len(METRICS)
            for svc in SERVICES
        }

        # Episode state
        self.global_state: Optional[np.ndarray] = None  # shape: (30,)
        self.step_count: int = 0

        print("MarlEnvironment initialized.")
        print(f"  DT         : {dt_path}")
        print(f"  Reward cfg : {reward_cfg_path}")
        print(f"  Max steps  : {MAX_STEPS}")
        print()
        self._print_obs_space()

    # Helper: Observation Space Info

    def _print_obs_space(self) -> None:
        print("  Observation Space (partial observability):")
        print(f"  {'Agent':<12} {'Observed Services':<35} {'Obs Size':>10}")
        print(f"  {'-'*60}")
        for svc in SERVICES:
            neighbors = AGENT_NEIGHBORS[svc]
            obs_list  = [svc] + neighbors
            dim       = self.obs_dims[svc]
            print(f"  {svc:<12} {str(obs_list):<35} {dim:>10}")

    # Extracting Partial Observations

    def _get_obs(self, global_state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract a partial observation for each agent from the (30,) global state.

        Each agent: its own 5 metrics + 5 metrics from each neighbor
        Example -> cart: [cart_5] + [catalogue_5] + [payment_5] = 15 dims

        Returns:
            Dict: {service_name: np.ndarray(obs_dim,)}
        """
        observations = {}
        for svc in SERVICES:
            # Get own metrics
            start = SERVICE_IDX[svc]
            own_metrics = global_state[start: start + len(METRICS)]

            # Get neighbor metrics
            neighbor_metrics = []
            for neighbor in AGENT_NEIGHBORS[svc]:
                n_start = SERVICE_IDX[neighbor]
                neighbor_metrics.append(
                    global_state[n_start: n_start + len(METRICS)]
                )

            # Concatenate: [own | neighbor1 | neighbor2 | ...]
            if neighbor_metrics:
                observations[svc] = np.concatenate(
                    [own_metrics] + neighbor_metrics
                ).astype(np.float32)
            else:
                observations[svc] = own_metrics.astype(np.float32)

        return observations

    # Reward Computation

    def _compute_rewards(
        self,
        global_state: np.ndarray,
        actions_internal: Dict[str, int],
    ) -> Dict[str, float]:

        reward = 0.0
        total_pods = 0

        for i, svc in enumerate(SERVICES):
            cfg = self.reward_cfg[svc]
            lat_idx = SERVICE_IDX[svc] + 4  # latency, 5th metric (index 4)

            latency = global_state[lat_idx]

            # Accumulate pod count (for the penalty)
            pod_idx = SERVICE_IDX[svc] + 0  # num_pods, 1st metric (index 0)
            total_pods += global_state[pod_idx]

            if not cfg.get('in_reward', True):
                continue

            if latency <= 0:
                continue


            p50    = cfg['p50']
            weight = cfg['weight']


            p50_norm = self._get_normalized_p50(svc)
            reward += np.tanh(1.0 - latency / p50_norm) * weight


        pod_cost = self.reward_cfg.get('cart', {}).get('pod_cost', 0.05)
        reward -= total_pods * 0.05

        return {svc: round(float(reward), 4) for svc in SERVICES}

    def _get_normalized_p50(self, svc: str) -> float:

        # Scaler feature order: all state_cols
        state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]
        lat_col    = f'{svc}_latency'

        if lat_col not in state_cols:
            return 1.0  # fallback

        col_idx = state_cols.index(lat_col)

        # MinMaxScaler: x_norm = (x - min) / (max - min)
        p50_raw   = self.reward_cfg[svc]['p50']
        data_min  = self.scaler.data_min_[col_idx]
        data_max  = self.scaler.data_max_[col_idx]

        if data_max == data_min:
            return 0.5  # edge case

        p50_norm = (p50_raw - data_min) / (data_max - data_min)
        return float(np.clip(p50_norm, 0.01, 1.0))  # avoid division by zero

    # reset()

    def reset(
        self, initial_state: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:

        if initial_state is not None:
            assert initial_state.shape == (30,), \
                f"initial_state shape must be (30,), got: {initial_state.shape}"
            self.global_state = initial_state.astype(np.float32).copy()
        else:

            self.global_state = np.full(30, 0.5, dtype=np.float32)

        self.step_count = 0
        return self._get_obs(self.global_state)

    # step() 

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:

        assert self.global_state is not None, "reset() was not called!"
        assert set(actions.keys()) == set(SERVICES), \
            f"Action required for all services. Missing: {set(SERVICES) - set(actions.keys())}"

        # 1. Build the action vector: {0,1,2} -> {-1,0,+1}
        action_vector = np.array(
            [ACTION_MAP[actions[svc]] for svc in SERVICES],
            dtype=np.float32
        )  # shape: (6,)

        # 2. Feed into the DT: [state(30) | action(6)] -> next_state(30)
        state_action = np.concatenate([self.global_state, action_vector])
        with torch.no_grad():
            inp        = torch.tensor(state_action, dtype=torch.float32).unsqueeze(0)
            self.dt.eval()
            next_state = self.dt(inp).squeeze(0).numpy()  # (30,)

        # 3. Compute rewards (based on next_state)
        rewards = self._compute_rewards(next_state, actions)

        # 4. Update the state
        self.global_state = next_state
        self.step_count  += 1

        # 5. Episode termination condition
        done = self.step_count >= MAX_STEPS

        # 6. Return partial observations
        observations = self._get_obs(next_state)

        # 7. Debug information
        info = {
            'step':          self.step_count,
            'global_state':  next_state.copy(),
            'action_vector': action_vector,           # internal: -1/0/+1 (fed to the DT)
            'actions':       actions,                 # discrete:  0/1/2  (agent output)
            'global_reward': list(rewards.values())[0],
        }

        return observations, rewards, done, info

    # Helper: Print Info

    def obs_dim(self, svc: str) -> int:
        """Return the observation size for the given service."""
        return self.obs_dims[svc]

    def action_dim(self) -> int:
        """Action space size (same for all agents)."""
        return N_ACTIONS

    def print_step_summary(self, step_info: Dict) -> None:
        """Print the step() output in a readable format (for debugging)."""
        # action_vector is internal (-1/0/+1) -- show with readable labels
        labels = {-1: 'down', 0: 'keep', 1: 'up'}
        action_display = {
            s: labels[int(a)]
            for s, a in zip(SERVICES, step_info['action_vector'])
        }
        print(f"\n  [Step {step_info['step']}]")
        print(f"  Actions  : {action_display}")
        print(f"  Reward   : {step_info['global_reward']:.4f}")

        gs = step_info['global_state']
        print(f"\n  {'Service':<12} {'pods':>6} {'cpu':>8} {'mem':>8} {'rps':>8} {'lat':>8}")
        print(f"  {'-'*55}")
        for svc in SERVICES:
            start = SERVICE_IDX[svc]
            vals  = gs[start: start + len(METRICS)]
            print(f"  {svc:<12} " + " ".join(f"{v:>8.4f}" for v in vals))


# Quick Validation (standalone run)

if __name__ == '__main__':
    print("=" * 65)
    print("MarlEnvironment — Quick Validation")
    print("=" * 65)

    # Initialize the environment
    env = MarlEnvironment(
        dt_path='digital_twin_best.pth',
        reward_cfg_path='reward_config.json',
        scaler_path='scaler.pkl',
    )

    print("\n" + "=" * 65)
    print("reset() test")
    print("=" * 65)

    # Use a real initial state from the SAR dataset
    import pandas as pd
    df = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]
    initial    = df.iloc[100][state_cols].values.astype(np.float32)

    obs = env.reset(initial_state=initial)

    print(f"\n  global_state shape : {env.global_state.shape}")
    for svc in SERVICES:
        print(f"  {svc:<12} obs shape: {obs[svc].shape}  "
              f"(expected: {env.obs_dim(svc)})")

    print("\n" + "=" * 65)
    print("step() test — 3 steps, all agents 'keep' (action 1)")
    print("=" * 65)

    for i in range(3):
        actions = {svc: 1 for svc in SERVICES}  # all keep
        obs, rewards, done, info = env.step(actions)
        env.print_step_summary(info)
        if done:
            print("  -> Episode finished!")
            break

    print("\n" + "=" * 65)
    print("step() test — action diversity")
    print("=" * 65)

    obs = env.reset(initial_state=initial)
    test_actions = {
        'cart':      2,   # scale up
        'catalogue': 1,   # keep
        'payment':   1,   # keep
        'shipping':  0,   # scale down
        'ratings':   1,   # keep
        'user':      2,   # scale up
    }
    obs, rewards, done, info = env.step(test_actions)
    env.print_step_summary(info)

    print("\n" + "=" * 65)
    print("Full Episode (10 steps)")
    print("=" * 65)

    import random
    obs      = env.reset(initial_state=initial)
    ep_reward = 0.0
    step_n    = 0

    while True:
        # Random actions (baseline test)
        actions = {svc: random.choice([0, 1, 2]) for svc in SERVICES}
        obs, rewards, done, info = env.step(actions)
        ep_reward += info['global_reward']
        step_n    += 1
        if done:
            break

    print(f"\n  Episode complete.")
    print(f"  Total steps   : {step_n}")
    print(f"  Total reward  : {ep_reward:.4f}")
    print(f"  Average reward: {ep_reward / step_n:.4f}")

    print("\n  OK MarlEnvironment validation successful!")
    print("=" * 65)
