import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Optional

HIDDEN_LOCAL  = [128, 64]
HIDDEN_MIX    = 64
EMBED_DIM     = 32
LR            = 5e-5
GAMMA         = 0.99
BUFFER_SIZE   = 10_000
BATCH_SIZE    = 64
TARGET_SYNC   = 10
WARMUP_STEPS  = 500
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 2000
N_AGENTS      = 6
GLOBAL_DIM    = 30
MAX_GRAD_NORM = 1.0


class LocalQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in HIDDEN_LOCAL:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs):
        return self.net(obs)


class MixingNetwork(nn.Module):
    def __init__(self, n_agents=N_AGENTS, global_dim=GLOBAL_DIM):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_dim, EMBED_DIM), nn.ReLU(),
            nn.Linear(EMBED_DIM, n_agents * HIDDEN_MIX),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_dim, EMBED_DIM), nn.ReLU(),
            nn.Linear(EMBED_DIM, HIDDEN_MIX),
        )
        self.hyper_b1 = nn.Linear(global_dim, HIDDEN_MIX)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_dim, EMBED_DIM), nn.ReLU(),
            nn.Linear(EMBED_DIM, 1),
        )

    def forward(self, q_values, global_state):
        batch = q_values.shape[0]
        w1    = torch.abs(self.hyper_w1(global_state)).view(batch, HIDDEN_MIX, self.n_agents)
        b1    = self.hyper_b1(global_state).view(batch, HIDDEN_MIX, 1)
        q     = q_values.unsqueeze(2)
        hidden = torch.nn.functional.elu(torch.bmm(w1, q) + b1)
        w2    = torch.abs(self.hyper_w2(global_state)).view(batch, 1, HIDDEN_MIX)
        b2    = self.hyper_b2(global_state).view(batch, 1, 1)
        return (torch.bmm(w2, hidden) + b2).squeeze(2)


class QMIXReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, global_state, next_gs, done):
        self.buffer.append((obs, actions, reward, next_obs, global_state, next_gs, done))

    def sample(self, batch_size, services):
        batch = random.sample(self.buffer, batch_size)
        obs_b, act_b, rew_b, nobs_b, gs_b, ngs_b, done_b = zip(*batch)
        result = {
            'global_states':      torch.tensor(np.array(gs_b),  dtype=torch.float32),
            'next_global_states': torch.tensor(np.array(ngs_b), dtype=torch.float32),
            'rewards': torch.tensor(np.array(rew_b, dtype=np.float32)).unsqueeze(1),
            'dones':   torch.tensor(np.array(done_b, dtype=np.float32)).unsqueeze(1),
        }
        for svc in services:
            result[f'obs_{svc}']  = torch.tensor(np.array([o[svc] for o in obs_b]),  dtype=torch.float32)
            result[f'nobs_{svc}'] = torch.tensor(np.array([o[svc] for o in nobs_b]), dtype=torch.float32)
            result[f'act_{svc}']  = torch.tensor(np.array([a[svc] for a in act_b]),  dtype=torch.long)
        return result

    def __len__(self):
        return len(self.buffer)

    def is_ready(self):
        return len(self.buffer) >= WARMUP_STEPS


class EpsilonScheduler:
    def __init__(self):
        self.step_count = 0

    def value(self):
        return EPS_START + min(self.step_count / EPS_DECAY, 1.0) * (EPS_END - EPS_START)

    def increment(self):
        self.step_count += 1


class QMIXSystem:
    def __init__(self, env, device='cpu'):
        from marl_env import SERVICES
        self.services = SERVICES
        self.device   = torch.device(device)

        self.q_nets    = {svc: LocalQNetwork(env.obs_dim(svc), env.action_dim()).to(self.device) for svc in SERVICES}
        self.q_targets = {svc: LocalQNetwork(env.obs_dim(svc), env.action_dim()).to(self.device) for svc in SERVICES}
        for svc in SERVICES:
            self.q_targets[svc].load_state_dict(self.q_nets[svc].state_dict())
            self.q_targets[svc].eval()

        self.mixer        = MixingNetwork().to(self.device)
        self.mixer_target = MixingNetwork().to(self.device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())
        self.mixer_target.eval()

        all_params = list(self.mixer.parameters())
        for svc in SERVICES:
            all_params += list(self.q_nets[svc].parameters())
        self.optimizer = optim.Adam(all_params, lr=LR)

        self.buffer        = QMIXReplayBuffer(BUFFER_SIZE)
        self.eps_sched     = EpsilonScheduler()
        self.total_steps   = 0
        self.total_updates = 0

    def select_actions(self, obs, training=True):
        actions = {}
        for svc in self.services:
            if training and random.random() < self.eps_sched.value():
                actions[svc] = random.randint(0, 2)
            else:
                obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    actions[svc] = int(self.q_nets[svc](obs_t).argmax(dim=-1).item())
        return actions

    def step(self):
        self.eps_sched.increment()
        self.total_steps += 1

    def update(self):
        if not self.buffer.is_ready():
            return None

        batch   = self.buffer.sample(BATCH_SIZE, self.services)
        gs      = batch['global_states'].to(self.device)
        ngs     = batch['next_global_states'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones   = batch['dones'].to(self.device)

        # Online Q of the taken actions
        q_vals = []
        for svc in self.services:
            obs_t = batch[f'obs_{svc}'].to(self.device)
            act_t = batch[f'act_{svc}'].to(self.device)
            q_a   = self.q_nets[svc](obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
            q_vals.append(q_a)
        qtot = self.mixer(torch.stack(q_vals, dim=1), gs)

        # Target Q
        with torch.no_grad():
            q_next = []
            for svc in self.services:
                nobs_t = batch[f'nobs_{svc}'].to(self.device)
                q_next.append(self.q_targets[svc](nobs_t).max(dim=1).values)
            qtot_target = self.mixer_target(torch.stack(q_next, dim=1), ngs)
            y = rewards + GAMMA * qtot_target * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(qtot, y)   # Huber loss, robust to outliers

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.mixer.parameters()) +
            [p for svc in self.services for p in self.q_nets[svc].parameters()],
            max_norm=MAX_GRAD_NORM
        )
        self.optimizer.step()
        self.total_updates += 1

        if self.total_updates % TARGET_SYNC == 0:
            for svc in self.services:
                self.q_targets[svc].load_state_dict(self.q_nets[svc].state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save({
            'q_nets':       {svc: self.q_nets[svc].state_dict()    for svc in self.services},
            'q_targets':    {svc: self.q_targets[svc].state_dict() for svc in self.services},
            'mixer':        self.mixer.state_dict(),
            'mixer_target': self.mixer_target.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'total_steps':  self.total_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for svc in self.services:
            self.q_nets[svc].load_state_dict(ckpt['q_nets'][svc])
            self.q_targets[svc].load_state_dict(ckpt['q_targets'][svc])
        self.mixer.load_state_dict(ckpt['mixer'])
        self.mixer_target.load_state_dict(ckpt['mixer_target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)

    def param_count(self):
        local = sum(sum(p.numel() for p in self.q_nets[svc].parameters()) for svc in self.services)
        mix   = sum(p.numel() for p in self.mixer.parameters())
        return {'local_q_total': local, 'mixing_network': mix}


if __name__ == '__main__':
    import pandas as pd
    from marl_env import MarlEnvironment, SERVICES

    print("=" * 65)
    print("QMIX v2 — Quick Check")
    print("=" * 65)
    print(f"  TARGET_SYNC : {TARGET_SYNC}  (v1: 50)")
    print(f"  LR          : {LR}   (v1: 1e-4)")
    print(f"  MAX_GRAD    : {MAX_GRAD_NORM}   (v1: 10.0)")
    print(f"  Loss        : Huber  (v1: MSE)")

    env  = MarlEnvironment()
    qmix = QMIXSystem(env)
    counts = qmix.param_count()
    print(f"\n  Local Q total  : {counts['local_q_total']:,}")
    print(f"  Mixing network : {counts['mixing_network']:,}")

    gs_t = torch.zeros(1, GLOBAL_DIM)
    with torch.no_grad():
        qtot_low = qmix.mixer(torch.zeros(1, N_AGENTS), gs_t).item()
        qtot_hi  = qmix.mixer(torch.ones(1, N_AGENTS),  gs_t).item()
    print(f"\n  Monotonicity: Q_low→{qtot_low:.4f}  Q_high→{qtot_hi:.4f}  "
          f"{'✓' if qtot_hi >= qtot_low else '✗'}")

    df         = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES
                  for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
    initial    = df.iloc[100][state_cols].values.astype(np.float32)

    obs = env.reset(initial_state=initial)
    print(f"\n  Warmup {WARMUP_STEPS} steps...", end='', flush=True)
    for _ in range(WARMUP_STEPS):
        gs      = env.global_state.copy()
        actions = qmix.select_actions(obs, training=True)
        nobs, rewards, done, info = env.step(actions)
        ngs = env.global_state.copy()
        qmix.buffer.push(obs, actions, info['global_reward'], nobs, gs, ngs, done)
        qmix.step()
        obs = nobs if not done else env.reset(initial_state=initial)
    print(" ✓")

    print("  First 5 updates:")
    for i in range(5):
        loss = qmix.update()
        print(f"    Update {i+1}: loss = {loss:.6f}")

    qmix.save('qmix_test.pth')
    qmix.load('qmix_test.pth')
    print("  save/load ✓")
    print("\n  ✓ QMIX v2 check passed!")
    print("=" * 65)
