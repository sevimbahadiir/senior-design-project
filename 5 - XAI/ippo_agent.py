import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Optional


# Hyperparameters

HIDDEN     = [128, 64]
LR_ACTOR   = 1e-4
LR_CRITIC  = 5e-4
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.2
ENTROPY_C  = 0.02  
VALUE_C    = 0.5
K_EPOCHS   = 8
MAX_GRAD   = 0.5


# Network Architectures

class ActorNet(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        layers = []
        prev = obs_dim
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
        prev = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# Rollout Buffer

class RolloutBuffer:
    def __init__(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.dones:     List[bool]       = []
        self.values:    List[float]      = []

    def push(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# GAE Computation

def compute_gae(
    rewards:    List[float],
    values:     List[float],
    dones:      List[bool],
    last_value: float = 0.0,
    gamma:      float = GAMMA,
    lam:        float = GAE_LAMBDA,
) -> Tuple[List[float], List[float]]:
    """Generalized Advantage Estimation (GAE-lambda)."""
    T          = len(rewards)
    advantages = [0.0] * T
    returns    = [0.0] * T
    gae        = 0.0

    for t in reversed(range(T)):
        next_val  = last_value if t == T - 1 else values[t + 1]
        next_done = 1.0 - float(dones[t])
        delta     = rewards[t] + gamma * next_val * next_done - values[t]
        gae       = delta + gamma * lam * next_done * gae
        advantages[t] = gae
        returns[t]    = gae + values[t]

    return advantages, returns


# IPPO Agent

class IPPOAgent:

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

        self.buffer = RolloutBuffer()
        self.total_updates = 0

    #  Action Selection

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        dist     = self.actor.get_dist(obs_t)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        value    = self.critic(obs_t).squeeze()
        return int(action_t.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return float(self.critic(obs_t).squeeze().item())

    #  Policy Update

    def update(
        self,
        last_obs:     Optional[np.ndarray] = None,
        entropy_coef: float = ENTROPY_C,       # <- NEW: passed dynamically by train_ippo_v4.py
    ) -> Dict[str, float]:

        if len(self.buffer) == 0:
            return {}

        last_value = 0.0
        if last_obs is not None:
            last_value = self.get_value(last_obs)

        advantages, returns = compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            last_value=last_value,
        )

        obs_t     = torch.tensor(np.array(self.buffer.obs),
                                 dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(self.buffer.actions,
                                 dtype=torch.long).to(self.device)
        old_lp_t  = torch.tensor(self.buffer.log_probs,
                                 dtype=torch.float32).to(self.device)
        adv_t     = torch.tensor(advantages,
                                 dtype=torch.float32).to(self.device)
        returns_t = torch.tensor(returns,
                                 dtype=torch.float32).to(self.device)

        if adv_t.shape[0] > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        metrics_acc = {'actor_loss': 0.0, 'critic_loss': 0.0,
                       'entropy': 0.0, 'approx_kl': 0.0}

        for epoch in range(K_EPOCHS):

            dist     = self.actor.get_dist(obs_t)
            new_lp   = dist.log_prob(actions_t)
            entropy  = dist.entropy().mean()

            ratio    = torch.exp(new_lp - old_lp_t)
            surr1    = ratio * adv_t
            surr2    = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t

            entropy_floored = torch.clamp(entropy, min=0.3)
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy_floored

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD)
            self.actor_opt.step()

            values      = self.critic(obs_t).squeeze()
            critic_loss = VALUE_C * nn.functional.mse_loss(values, returns_t)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD)
            self.critic_opt.step()

            with torch.no_grad():
                approx_kl = ((old_lp_t - new_lp) ** 2).mean() * 0.5

            metrics_acc['actor_loss']  += actor_loss.item()
            metrics_acc['critic_loss'] += critic_loss.item()
            metrics_acc['entropy']     += entropy.item()
            metrics_acc['approx_kl']   += approx_kl.item()

        for k in metrics_acc:
            metrics_acc[k] /= K_EPOCHS

        self.total_updates += 1
        return metrics_acc

    # ─ Save / Load Model 

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
        return (f"IPPOAgent(service={self.service}, obs_dim={self.obs_dim}, "
                f"actor={ap:,}, critic={cp:,})")


# Agent Pool

def build_agent_pool(env, device: str = 'cpu') -> Dict[str, 'IPPOAgent']:
    """Create an IPPOAgent for each service based on the MarlEnvironment."""
    from marl_env import SERVICES
    agents = {}
    for svc in SERVICES:
        agents[svc] = IPPOAgent(
            obs_dim=env.obs_dim(svc),
            n_actions=env.action_dim(),
            service=svc,
            device=device,
        )
    return agents


# Quick Validation

if __name__ == '__main__':
    import pandas as pd
    from marl_env import MarlEnvironment, SERVICES

    print("=" * 65)
    print("IPPOAgent v3 — Quick Validation")
    print("=" * 65)

    env    = MarlEnvironment()
    agents = build_agent_pool(env)

    print("\nAgent Pool:")
    print(f"  {'Service':<12} {'Obs Dim':>8} {'Actor':>12} {'Critic':>12}")
    print(f"  {'-'*48}")
    for svc, agent in agents.items():
        ap = sum(p.numel() for p in agent.actor.parameters())
        cp = sum(p.numel() for p in agent.critic.parameters())
        print(f"  {svc:<12} {agent.obs_dim:>8} {ap:>12,} {cp:>12,}")

    df         = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES
                  for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
    initial    = df.iloc[100][state_cols].values.astype('float32')

    obs = env.reset(initial_state=initial)
    ep_reward = 0.0
    last_obs  = obs.copy()

    while True:
        actions = {}
        for svc in SERVICES:
            action, log_prob, value = agents[svc].select_action(obs[svc])
            actions[svc] = action
            agents[svc].buffer.push(obs[svc], action, log_prob, 0.0, False, value)

        obs, rewards, done, info = env.step(actions)
        ep_reward += info['global_reward']
        for svc in SERVICES:
            agents[svc].buffer.rewards[-1] = rewards[svc]
            agents[svc].buffer.dones[-1]   = done
        last_obs = obs.copy()
        if done:
            break

    print(f"\n  Episode reward: {ep_reward:.4f}")

    test_entropy = 0.035
    print(f"\n  update() test (entropy_coef={test_entropy}):")
    print(f"  {'Service':<12} {'ActorLoss':>11} {'Entropy':>9}")
    print(f"  {'-'*36}")
    for svc in SERVICES:
        m = agents[svc].update(last_obs=last_obs[svc], entropy_coef=test_entropy)
        agents[svc].buffer.clear()
        print(f"  {svc:<12} {m['actor_loss']:>11.4f} {m['entropy']:>9.4f}")

    print("\n  OK dynamic entropy_coef passing works")
    print("  OK IPPOAgent v3 validation successful!")
    print("=" * 65)
