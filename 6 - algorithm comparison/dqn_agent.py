import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Tuple, Optional, List

# ─── Hyperparameters ─────────────────────────────────────────────────────────

HIDDEN       = [128, 64]
LR           = 1e-4
GAMMA        = 0.99
BUFFER_SIZE  = 10_000
BATCH_SIZE   = 64
TARGET_SYNC  = 50       # in updates, not episodes
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 2000
WARMUP_STEPS = 500


# ─── Q-Network ───────────────────────────────────────────────────────────────

class QNetwork(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        layers = []
        prev   = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → Q values  (shape: [batch, n_actions])"""
        return self.net(obs)


# ─── Replay Buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a random batch."""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Has the warmup finished?"""
        return len(self.buffer) >= WARMUP_STEPS


# ─── Epsilon Scheduler ───────────────────────────────────────────────────────

class EpsilonScheduler:

    def __init__(self):
        self.step_count = 0

    def value(self) -> float:
        progress = min(self.step_count / EPS_DECAY, 1.0)
        return EPS_START + progress * (EPS_END - EPS_START)

    def increment(self) -> None:
        self.step_count += 1


# ─── DQN Agent ───────────────────────────────────────────────────────────────

class DQNAgent:


    def __init__(
        self,
        obs_dim:   int,
        n_actions: int = 3,
        service:   str = 'unknown',
        device:    str = 'cpu',
    ):
        self.service   = service
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.device    = torch.device(device)

        # Online network: updated every step
        self.q_net     = QNetwork(obs_dim, n_actions).to(self.device)

        # Target network: synced every TARGET_SYNC updates to provide a
        # stable target while the online network keeps changing
        self.q_target  = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)
        self.eps_sched = EpsilonScheduler()

        self.total_steps   = 0
        self.total_updates = 0

    # ── Action Selection ─────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:

        if training and random.random() < self.eps_sched.value():
            return random.randint(0, self.n_actions - 1)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        return int(q_vals.argmax(dim=-1).item())

    # ── Bellman Update ───────────────────────────────────────────────────────

    def update(self) -> Optional[float]:

        if not self.buffer.is_ready():
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(BATCH_SIZE)

        obs_t      = torch.tensor(obs,     dtype=torch.float32).to(self.device)
        actions_t  = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t  = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(next_obs,dtype=torch.float32).to(self.device)
        dones_t    = torch.tensor(dones,   dtype=torch.float32).to(self.device)

        # Q_online(s, a) for the taken action only
        q_current = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # r + γ · max Q_target(s', ·)
        with torch.no_grad():
            q_next   = self.q_target(next_obs_t).max(dim=1).values
            q_target = rewards_t + GAMMA * q_next * (1.0 - dones_t)

        # Huber loss (more robust to outliers than MSE)
        loss = nn.functional.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping keeps Q values from exploding
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_updates += 1

        if self.total_updates % TARGET_SYNC == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def step(self) -> None:
        """Advance the epsilon scheduler (called every step)."""
        self.eps_sched.increment()
        self.total_steps += 1

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            'q_net':       self.q_net.state_dict(),
            'q_target':    self.q_target.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'obs_dim':     self.obs_dim,
            'n_actions':   self.n_actions,
            'service':     self.service,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (f"DQNAgent(service={self.service}, "
                f"obs_dim={self.obs_dim}, params={params:,}, "
                f"eps={self.eps_sched.value():.3f})")


# ─── Agent Pool ──────────────────────────────────────────────────────────────

def build_dqn_pool(
    env,
    device: str = 'cpu',
) -> Dict[str, DQNAgent]:
    """
    Create one DQNAgent per service for the given MarlEnvironment.
    """
    from marl_env import SERVICES
    agents = {}
    for svc in SERVICES:
        agents[svc] = DQNAgent(
            obs_dim   = env.obs_dim(svc),
            n_actions = env.action_dim(),
            service   = svc,
            device    = device,
        )
    return agents


# ─── Quick Check ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd
    from marl_env import MarlEnvironment, SERVICES

    print("=" * 65)
    print("DQNAgent — Quick Check")
    print("=" * 65)

    env    = MarlEnvironment()
    agents = build_dqn_pool(env)

    print("\nAgent Pool:")
    print(f"  {'Service':<12} {'Obs Dim':>8} {'Q-Net Params':>14} {'Buffer':>8}")
    print(f"  {'-'*46}")
    for svc, agent in agents.items():
        p = sum(p.numel() for p in agent.q_net.parameters())
        print(f"  {svc:<12} {agent.obs_dim:>8} {p:>14,} {len(agent.buffer):>8}")

    print("\n" + "=" * 65)
    print("Epsilon decay test")
    print("=" * 65)
    test_agent = agents['cart']
    for step in [0, 500, 1000, 2000, 3000]:
        test_agent.eps_sched.step_count = step
        print(f"  Step {step:>5}: ε = {test_agent.eps_sched.value():.4f}")

    print("\n" + "=" * 65)
    print("select_action() + buffer + update() test")
    print("=" * 65)

    df         = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES
                  for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
    initial    = df.iloc[100][state_cols].values.astype(np.float32)

    print(f"\n  Warmup: filling the buffer with {WARMUP_STEPS} transitions...")
    obs = env.reset(initial_state=initial)
    warmup_steps = 0
    while warmup_steps < WARMUP_STEPS:
        actions = {svc: agents[svc].select_action(obs[svc]) for svc in SERVICES}
        next_obs, rewards, done, info = env.step(actions)
        for svc in SERVICES:
            agents[svc].buffer.push(
                obs[svc], actions[svc],
                rewards[svc], next_obs[svc], done
            )
            agents[svc].step()
        obs = next_obs if not done else env.reset(initial_state=initial)
        warmup_steps += 1

    print(f"  Buffer size    : {len(agents['cart'].buffer)}")
    print(f"  Current epsilon: {agents['cart'].eps_sched.value():.4f}")

    print(f"\n  First 5 updates (cart agent):")
    for i in range(5):
        loss = agents['cart'].update()
        print(f"    Update {i+1}: loss = {loss:.6f}")

    print(f"\n  Target sync: every {TARGET_SYNC} updates")
    print(f"  Current update count: {agents['cart'].total_updates}")

    agents['cart'].save('cart_dqn_test.pth')
    agents['cart'].load('cart_dqn_test.pth')
    print(f"\n  save/load ✓")

    print("\n  ✓ DQNAgent check passed!")
    print("=" * 65)
