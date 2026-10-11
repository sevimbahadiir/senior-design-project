
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
from a2c_agent import A2CAgent, build_a2c_pool

# ─── Hyperparameters ─────────────────────────────────────────────────────────

N_EPISODES     = 3000
SAVE_INTERVAL  = 500
LOG_INTERVAL   = 50
SEED           = 42
CHECKPOINT_DIR = 'checkpoints_a2c'
REWARD_NORM_EPS = 1e-8

# ─── Running Normalizer ──────────────────────────────────────────────────────

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
agents     = build_a2c_pool(env)
normalizer = RunningNormalizer()

print("=" * 65)
print("A2C Training")
print("=" * 65)
print(f"  Episodes    : {N_EPISODES}")
print(f"  Update      : every step (instantaneous TD)")
print(f"  Reward norm : RunningNormalizer")
print(f"  LR a/c      : 1e-4 / 5e-4")
print(f"  Entropy C   : 0.02  |  Value C: 0.5")
print("=" * 65)

df_sar     = pd.read_csv('rl_dataset_sar_normalized.csv')
state_cols = [f'{s}_{m}' for s in SERVICES
              for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
initial_states = df_sar[state_cols].values.astype(np.float32)
print(f"\n  {len(initial_states)} initial states loaded.\n")

# ─── Training Loop ───────────────────────────────────────────────────────────

log         = defaultdict(list)
train_start = time.time()

print("Starting training...\n")

for episode in range(1, N_EPISODES + 1):

    idx  = random.randint(0, len(initial_states) - 1)
    obs  = env.reset(initial_state=initial_states[idx])
    ep_raw      = 0.0
    ep_al = ep_cl = ep_en = ep_adv = 0.0
    steps = 0

    while True:
        actions = {}
        for svc in SERVICES:
            action, _, _ = agents[svc].select_action(obs[svc])
            actions[svc] = action

        next_obs, rewards, done, info = env.step(actions)

        raw_r = info['global_reward']
        normalizer.update(raw_r)
        norm_r = normalizer.normalize(raw_r)

        for svc in SERVICES:
            m = agents[svc].update(
                obs      = obs[svc],
                action   = actions[svc],
                reward   = norm_r,
                next_obs = next_obs[svc],
                done     = done,
            )
            ep_al  += m['actor_loss']
            ep_cl  += m['critic_loss']
            ep_en  += m['entropy']
            ep_adv += m['advantage']

        ep_raw += raw_r
        steps  += 1
        obs     = next_obs
        if done:
            break

    # Per-step averages
    n = steps * len(SERVICES)
    log['episode'].append(episode)
    log['reward_raw'].append(ep_raw)
    log['actor_loss'].append(ep_al / n)
    log['critic_loss'].append(ep_cl / n)
    log['entropy'].append(ep_en / n)
    log['advantage'].append(ep_adv / n)

    if episode % LOG_INTERVAL == 0 or episode == 1:
        elapsed = time.time() - train_start
        eta     = (N_EPISODES - episode) / (episode / elapsed) / 60
        w       = min(LOG_INTERVAL, episode)
        avg_r   = np.mean(log['reward_raw'][-w:])
        avg_al  = np.mean(log['actor_loss'][-w:])
        avg_cl  = np.mean(log['critic_loss'][-w:])
        avg_en  = np.mean(log['entropy'][-w:])
        print(f"  Ep {episode:>5}/{N_EPISODES} | "
              f"Reward: {avg_r:>8.3f} | "
              f"ActorL: {avg_al:>8.4f} | "
              f"CriticL: {avg_cl:>6.4f} | "
              f"Entropy: {avg_en:>6.4f} | "
              f"ETA: {eta:.0f}min")

    if episode % SAVE_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'a2c_{episode}.pth')
        torch.save({
            'episode':    episode,
            'agents':     {svc: {
                'actor':      agents[svc].actor.state_dict(),
                'critic':     agents[svc].critic.state_dict(),
                'actor_opt':  agents[svc].actor_opt.state_dict(),
                'critic_opt': agents[svc].critic_opt.state_dict(),
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
print("A2C training complete.")
print(f"  Duration     : {elapsed/60:.1f} min")
print(f"  First 100 ep : {first100:.3f}  →  Last 100 ep: {last100:.3f}  "
      f"({last100-first100:+.3f})")
print(f"  Final entropy: {np.mean(log['entropy'][-100:]):.4f}")
print("=" * 65)

# ─── CSV + Plots ─────────────────────────────────────────────────────────────

pd.DataFrame(log).to_csv('training_log_a2c.csv', index=False)
print("\n  ✓ training_log_a2c.csv saved.")

def smooth(data, w=50):
    if len(data) < w: return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('A2C Training — Robot Shop MARL', fontsize=13, fontweight='bold')
eps_arr = np.array(log['episode'])

ax = axes[0]
ax.plot(eps_arr, log['reward_raw'], alpha=0.15, color='mediumpurple', lw=0.8)
if len(log['reward_raw']) >= 50:
    ax.plot(eps_arr[49:], smooth(log['reward_raw']), color='mediumpurple', lw=2)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_title('Episode Reward')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(eps_arr, log['critic_loss'], alpha=0.15, color='crimson', lw=0.8)
if len(log['critic_loss']) >= 50:
    ax.plot(eps_arr[49:], smooth(log['critic_loss']), color='crimson', lw=2)
ax.set_title('Critic Loss (TD error)')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax = axes[2]
ax.plot(eps_arr, log['entropy'], alpha=0.15, color='seagreen', lw=0.8)
if len(log['entropy']) >= 50:
    ax.plot(eps_arr[49:], smooth(log['entropy']), color='seagreen', lw=2)
ax.axhline(np.log(3), color='gray', ls='--', alpha=0.7,
           label=f'Uniform {np.log(3):.3f}')
ax.set_title('Policy Entropy')
ax.set_xlabel('Episode')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_plot_a2c.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ training_plot_a2c.png saved.")

torch.save({
    'episode': N_EPISODES,
    'agents':  {svc: {
        'actor':      agents[svc].actor.state_dict(),
        'critic':     agents[svc].critic.state_dict(),
        'actor_opt':  agents[svc].actor_opt.state_dict(),
        'critic_opt': agents[svc].critic_opt.state_dict(),
    } for svc in SERVICES},
    'normalizer': vars(normalizer),
    'log':        dict(log),
}, os.path.join(CHECKPOINT_DIR, 'a2c_final.pth'))

print(f"  ✓ Final model: {CHECKPOINT_DIR}/a2c_final.pth")
print("\nDone!")
