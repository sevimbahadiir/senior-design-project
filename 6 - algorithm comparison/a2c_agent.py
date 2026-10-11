import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional

# ─── Hyperparameters ─────────────────────────────────────────────────────────

HIDDEN    = [128, 64]
LR_ACTOR  = 1e-4
LR_CRITIC = 5e-4
GAMMA     = 0.99
ENTROPY_C = 0.02
VALUE_C   = 0.5
MAX_GRAD  = 0.5


# ─── Networks (same as IPPO) ─────────────────────────────────────────────────

class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        layers = []
        prev   = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        layers = []
        prev   = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─── A2C Agent ───────────────────────────────────────────────────────────────

class A2CAgent:


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

        self.actor  = ActorNet(obs_dim, n_actions).to(self.device)
        self.critic = CriticNet(obs_dim).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.total_updates = 0

    # ── Action Selection ─────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Sample an action from the policy.

        Returns:
            action   : int
            log_prob : float
            value    : float
        """
        obs_t    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        dist     = self.actor.get_dist(obs_t)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        value    = self.critic(obs_t).squeeze()
        return int(action_t.item()), float(log_prob.item()), float(value.item())

    # ── Per-step Update ──────────────────────────────────────────────────────

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Dict[str, float]:
        """
        Update actor and critic from a single transition (s, a, r, s', done).

        Returns:
            metrics: {'actor_loss', 'critic_loss', 'entropy', 'advantage'}
        """
        obs_t      = torch.tensor(obs,      dtype=torch.float32).unsqueeze(0).to(self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_t   = torch.tensor([action], dtype=torch.long).to(self.device)

        value      = self.critic(obs_t).squeeze()          # V(s)
        with torch.no_grad():
            next_value = self.critic(next_obs_t).squeeze() # V(s'), no grad for the target

        # A(s,a) = r + γ·V(s')·(1-done) - V(s)
        td_target  = reward + GAMMA * next_value * (1.0 - float(done))
        advantage  = td_target - value.detach()  # fixed target for the actor

        # ── Actor ──
        dist       = self.actor.get_dist(obs_t)
        log_prob   = dist.log_prob(action_t.squeeze())
        entropy    = dist.entropy().mean()
        actor_loss = -(log_prob * advantage) - ENTROPY_C * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD)
        self.actor_opt.step()

        # ── Critic ──
        critic_loss = VALUE_C * nn.functional.mse_loss(value, td_target.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD)
        self.critic_opt.step()

        self.total_updates += 1

        return {
            'actor_loss':  float(actor_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'entropy':     float(entropy.item()),
            'advantage':   float(advantage.item()),
        }

    # ── Deterministic Action (evaluation) ────────────────────────────────────

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        """Select the most likely action (evaluation mode)."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return int(self.actor(obs_t).argmax(dim=-1).item())

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            'actor':         self.actor.state_dict(),
            'critic':        self.critic.state_dict(),
            'actor_opt':     self.actor_opt.state_dict(),
            'critic_opt':    self.critic_opt.state_dict(),
            'obs_dim':       self.obs_dim,
            'n_actions':     self.n_actions,
            'service':       self.service,
            'total_updates': self.total_updates,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.total_updates = ckpt.get('total_updates', 0)

    def __repr__(self) -> str:
        ap = sum(p.numel() for p in self.actor.parameters())
        cp = sum(p.numel() for p in self.critic.parameters())
        return (f"A2CAgent(service={self.service}, "
                f"obs_dim={self.obs_dim}, "
                f"actor={ap:,}, critic={cp:,})")


# ─── Agent Pool ──────────────────────────────────────────────────────────────

def build_a2c_pool(env, device: str = 'cpu') -> Dict[str, A2CAgent]:
    from marl_env import SERVICES
    return {
        svc: A2CAgent(
            obs_dim   = env.obs_dim(svc),
            n_actions = env.action_dim(),
            service   = svc,
            device    = device,
        )
        for svc in SERVICES
    }


# ─── Quick Check ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd
    from marl_env import MarlEnvironment, SERVICES

    print("=" * 65)
    print("A2CAgent — Quick Check")
    print("=" * 65)

    env    = MarlEnvironment()
    agents = build_a2c_pool(env)

    print("\nAgent Pool:")
    print(f"  {'Service':<12} {'Obs Dim':>8} {'Actor':>10} {'Critic':>10}")
    print(f"  {'-'*44}")
    for svc, agent in agents.items():
        ap = sum(p.numel() for p in agent.actor.parameters())
        cp = sum(p.numel() for p in agent.critic.parameters())
        print(f"  {svc:<12} {agent.obs_dim:>8} {ap:>10,} {cp:>10,}")

    print("\n" + "=" * 65)
    print("select_action() + update() test — 1 episode")
    print("=" * 65)

    df         = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES
                  for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
    initial    = df.iloc[100][state_cols].values.astype(np.float32)

    obs           = env.reset(initial_state=initial)
    ep_reward     = 0.0
    step_metrics  = []

    while True:
        actions = {}
        log_probs = {}
        values    = {}

        for svc in SERVICES:
            action, lp, val = agents[svc].select_action(obs[svc])
            actions[svc]   = action
            log_probs[svc] = lp
            values[svc]    = val

        next_obs, rewards, done, info = env.step(actions)
        ep_reward += info['global_reward']

        for svc in SERVICES:
            m = agents[svc].update(
                obs      = obs[svc],
                action   = actions[svc],
                reward   = rewards[svc],
                next_obs = next_obs[svc],
                done     = done,
            )
            step_metrics.append(m)

        obs = next_obs
        if done:
            break

    print(f"\n  Episode reward : {ep_reward:.4f}")
    print(f"\n  Last-step metrics (cart):")
    last = step_metrics[-6]   # cart, last step
    for k, v in last.items():
        print(f"    {k:<15}: {v:.6f}")

    agents['cart'].save('cart_a2c_test.pth')
    agents['cart'].load('cart_a2c_test.pth')
    print(f"\n  save/load ✓")
    print("\n  ✓ A2CAgent check passed!")
    print("=" * 65)
