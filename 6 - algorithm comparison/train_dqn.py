

import os
import random
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from marl_env import MarlEnvironment, SERVICES
from dqn_agent import DQNAgent, build_dqn_pool, WARMUP_STEPS

# ─── Hyperparameters ─────────────────────────────────────────────────────────

N_EPISODES     = 3000
SAVE_INTERVAL  = 500
LOG_INTERVAL   = 50
SEED           = 42
CHECKPOINT_DIR = 'checkpoints_dqn'
REWARD_NORM_EPS = 1e-8

# ─── Running Normalizer (same as IPPO v2) ────────────────────────────────────

class RunningNormalizer:
    def __init__(self):
        self.mean = 0.0; self.var = 1.0; self.count = 0

    def update(self, v):
        self.count += 1
        d = v - self.mean; self.mean += d / self.count
        self.var += (v - self.mean) * d / self.count - self.var / self.count

    def normalize(self, v):
        return (v - self.mean) / (np.sqrt(max(self.var, REWARD_NORM_EPS)) + REWARD_NORM_EPS)

# ─── Setup ───────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env        = MarlEnvironment(seed=SEED)
agents     = build_dqn_pool(env)
normalizer = RunningNormalizer()

print("=" * 65)
print("DQN Training")
print("=" * 65)
print(f"  Episodes       : {N_EPISODES}")
print(f"  Warmup steps   : {WARMUP_STEPS}")
print(f"  Update/step    : every step (after warmup)")
print(f"  Reward norm    : RunningNormalizer")
print(f"  LR             : 1e-4  |  γ: 0.99")
print(f"  Buffer         : 10,000  |  Batch: 64")
print(f"  Target sync    : every 50 updates")
print("=" * 65)

# ─── SAR Dataset ─────────────────────────────────────────────────────────────

df_sar     = pd.read_csv('rl_dataset_sar_normalized.csv')
state_cols = [f'{s}_{m}' for s in SERVICES
              for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
initial_states = df_sar[state_cols].values.astype(np.float32)
print(f"\n  {len(initial_states)} initial states loaded.")

# ─── Warmup ──────────────────────────────────────────────────────────────────

print(f"  Warmup: filling the buffer for {WARMUP_STEPS} steps...", end='', flush=True)
obs = env.reset(initial_state=initial_states[0])
for _ in range(WARMUP_STEPS):
    actions  = {svc: agents[svc].select_action(obs[svc], training=True)
                for svc in SERVICES}
    next_obs, rewards, done, info = env.step(actions)
    raw_r = info['global_reward']
    normalizer.update(raw_r)
    norm_r = normalizer.normalize(raw_r)
    for svc in SERVICES:
        agents[svc].buffer.push(obs[svc], actions[svc], norm_r, next_obs[svc], done)
        agents[svc].step()
    obs = next_obs if not done else env.reset(
        initial_state=initial_states[random.randint(0, len(initial_states)-1)])
print(" ✓\n")

# ─── Training Loop ───────────────────────────────────────────────────────────

print("Starting training...\n")
log          = defaultdict(list)
train_start  = time.time()
global_step  = 0

for episode in range(1, N_EPISODES + 1):

    idx  = random.randint(0, len(initial_states) - 1)
    obs  = env.reset(initial_state=initial_states[idx])
    ep_raw  = 0.0
    ep_loss = []

    while True:
        actions = {svc: agents[svc].select_action(obs[svc], training=True)
                   for svc in SERVICES}

        next_obs, rewards, done, info = env.step(actions)
        global_step += 1

        raw_r = info['global_reward']
        normalizer.update(raw_r)
        norm_r = normalizer.normalize(raw_r)

        for svc in SERVICES:
            agents[svc].buffer.push(obs[svc], actions[svc], norm_r, next_obs[svc], done)
            agents[svc].step()

        step_losses = []
        for svc in SERVICES:
            loss = agents[svc].update()
            if loss is not None:
                step_losses.append(loss)

        if step_losses:
            ep_loss.append(np.mean(step_losses))

        ep_raw += raw_r
        obs     = next_obs
        if done:
            break

    log['episode'].append(episode)
    log['reward_raw'].append(ep_raw)
    log['epsilon'].append(agents['cart'].eps_sched.value())
    log['loss'].append(np.mean(ep_loss) if ep_loss else 0.0)

    if episode % LOG_INTERVAL == 0 or episode == 1:
        elapsed = time.time() - train_start
        eta     = (N_EPISODES - episode) / (episode / elapsed) / 60
        w       = min(LOG_INTERVAL, episode)
        avg_r   = np.mean(log['reward_raw'][-w:])
        avg_l   = np.mean(log['loss'][-w:])
        eps_val = log['epsilon'][-1]
        print(f"  Ep {episode:>5}/{N_EPISODES} | "
              f"Reward: {avg_r:>8.3f} | "
              f"Loss: {avg_l:>8.4f} | "
              f"ε: {eps_val:.3f} | "
              f"Steps: {global_step:>6} | "
              f"ETA: {eta:.0f}min")

    if episode % SAVE_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'dqn_{episode}.pth')
        torch.save({
            'episode': episode,
            'agents':  {svc: {
                'q_net':     agents[svc].q_net.state_dict(),
                'q_target':  agents[svc].q_target.state_dict(),
                'optimizer': agents[svc].optimizer.state_dict(),
            } for svc in SERVICES},
            'normalizer': vars(normalizer),
            'log':        dict(log),
        }, path)
        print(f"\n  ✓ Checkpoint: {path}\n")

# ─── End of Training ─────────────────────────────────────────────────────────

elapsed  = time.time() - train_start
first100 = np.mean(log['reward_raw'][:100])
last100  = np.mean(log['reward_raw'][-100:])

print("\n" + "=" * 65)
print("DQN training complete.")
print(f"  Duration     : {elapsed/60:.1f} min")
print(f"  Global steps : {global_step:,}")
print(f"  First 100 ep : {first100:.3f}  →  Last 100 ep: {last100:.3f}  "
      f"({last100-first100:+.3f})")
print(f"  Final ε      : {agents['cart'].eps_sched.value():.4f}")
print("=" * 65)

# ─── CSV ─────────────────────────────────────────────────────────────────────

pd.DataFrame(log).to_csv('training_log_dqn.csv', index=False)
print("\n  ✓ training_log_dqn.csv saved.")

# ─── Plots ───────────────────────────────────────────────────────────────────

def smooth(data, w=50):
    if len(data) < w:
        return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('DQN Training — Robot Shop MARL', fontsize=13, fontweight='bold')

eps_arr = np.array(log['episode'])

ax = axes[0]
ax.plot(eps_arr, log['reward_raw'], alpha=0.15, color='darkorange', lw=0.8)
if len(log['reward_raw']) >= 50:
    ax.plot(eps_arr[49:], smooth(log['reward_raw']), color='darkorange', lw=2)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_title('Episode Reward')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(eps_arr, log['loss'], alpha=0.15, color='crimson', lw=0.8)
if len(log['loss']) >= 50:
    ax.plot(eps_arr[49:], smooth(log['loss']), color='crimson', lw=2)
ax.set_title('Bellman Loss (should decrease)')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(eps_arr, log['epsilon'], color='steelblue', lw=2)
ax.set_title('Epsilon (exploration → exploitation)')
ax.set_xlabel('Episode')
ax.set_ylabel('ε')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_plot_dqn.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ training_plot_dqn.png saved.")

# ─── Final Model ─────────────────────────────────────────────────────────────

torch.save({
    'episode': N_EPISODES,
    'agents':  {svc: {
        'q_net':     agents[svc].q_net.state_dict(),
        'q_target':  agents[svc].q_target.state_dict(),
        'optimizer': agents[svc].optimizer.state_dict(),
    } for svc in SERVICES},
    'normalizer': vars(normalizer),
    'log':        dict(log),
}, os.path.join(CHECKPOINT_DIR, 'dqn_final.pth'))

print(f"  ✓ Final model: {CHECKPOINT_DIR}/dqn_final.pth")
print("\nDone!")
