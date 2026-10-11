
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
from mappo_agent import MAPPOAgent, build_mappo_pool

# ─── Hyperparameters ─────────────────────────────────────────────────────────

N_EPISODES          = 3000
EPISODES_PER_UPDATE = 5
SAVE_INTERVAL       = 500
LOG_INTERVAL        = 50
SEED                = 42
CHECKPOINT_DIR      = 'checkpoints_mappo'
REWARD_NORM_EPS     = 1e-8

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
agents     = build_mappo_pool(env)
normalizer = RunningNormalizer()

print("=" * 65)
print("MAPPO Training")
print("=" * 65)
print(f"  Episodes/update : {EPISODES_PER_UPDATE} ep = {EPISODES_PER_UPDATE*10} transitions")
print(f"  Critic input    : Global State (30 dims) — CTDE")
print(f"  Actor input     : Partial Obs (10-15 dims)")
print(f"  Reward norm     : RunningNormalizer")
print(f"  LR a/c          : 1e-4 / 5e-4  |  K_epochs: 8")
print("=" * 65)

df_sar     = pd.read_csv('rl_dataset_sar_normalized.csv')
state_cols = [f'{s}_{m}' for s in SERVICES
              for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
initial_states = df_sar[state_cols].values.astype(np.float32)
print(f"\n  {len(initial_states)} initial states loaded.\n")

# ─── Training Loop ───────────────────────────────────────────────────────────

log          = defaultdict(list)
train_start  = time.time()
update_count = 0

print("Starting training...\n")

for episode in range(1, N_EPISODES + 1):

    idx  = random.randint(0, len(initial_states) - 1)
    obs  = env.reset(initial_state=initial_states[idx])
    ep_raw = ep_norm = 0.0

    while True:
        # Global state for the critic
        global_state = env.global_state.copy()

        actions = {}
        for svc in SERVICES:
            action, log_prob, value = agents[svc].select_action(
                obs[svc], global_state
            )
            actions[svc] = action
            agents[svc].buffer.push(
                obs=obs[svc], global_state=global_state,
                action=action, log_prob=log_prob,
                reward=0.0, done=False, value=value,
            )

        next_obs, rewards, done, info = env.step(actions)

        raw_r = info['global_reward']
        normalizer.update(raw_r)
        norm_r = normalizer.normalize(raw_r)

        for svc in SERVICES:
            agents[svc].buffer.rewards[-1] = norm_r
            agents[svc].buffer.dones[-1]   = done

        ep_raw  += raw_r
        ep_norm += norm_r
        obs      = next_obs
        if done:
            break

    log['episode'].append(episode)
    log['reward_raw'].append(ep_raw)
    log['reward_norm'].append(ep_norm)

    if episode % EPISODES_PER_UPDATE == 0:
        update_count += 1
        last_gs = env.global_state.copy()
        ms = {}
        for svc in SERVICES:
            ms[svc] = agents[svc].update(last_global_state=last_gs)
            agents[svc].buffer.clear()

        log['update_ep'].append(episode)
        log['avg_actor_loss'].append(np.mean([ms[s]['actor_loss']  for s in SERVICES]))
        log['avg_critic_loss'].append(np.mean([ms[s]['critic_loss'] for s in SERVICES]))
        log['avg_entropy'].append(np.mean([ms[s]['entropy']     for s in SERVICES]))
        log['avg_kl'].append(np.mean([ms[s]['approx_kl']  for s in SERVICES]))

    if episode % LOG_INTERVAL == 0 or episode == 1:
        elapsed = time.time() - train_start
        eta     = (N_EPISODES - episode) / (episode / elapsed) / 60
        w       = min(LOG_INTERVAL, episode)
        avg_r   = np.mean(log['reward_raw'][-w:])
        al = log['avg_actor_loss'][-1]  if log['avg_actor_loss']  else 0
        cl = log['avg_critic_loss'][-1] if log['avg_critic_loss'] else 0
        en = log['avg_entropy'][-1]     if log['avg_entropy']     else 0
        print(f"  Ep {episode:>5}/{N_EPISODES} | "
              f"Reward: {avg_r:>8.3f} | "
              f"ActorL: {al:>8.4f} | "
              f"CriticL: {cl:>7.3f} | "
              f"Entropy: {en:>6.4f} | "
              f"ETA: {eta:.0f}min")

    if episode % SAVE_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'mappo_{episode}.pth')
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
print("MAPPO training complete.")
print(f"  Duration     : {elapsed/60:.1f} min  |  Updates: {update_count}")
print(f"  First 100 ep : {first100:.3f}  →  Last 100 ep: {last100:.3f}  "
      f"({last100-first100:+.3f})")
if log['avg_critic_loss']:
    print(f"  CriticLoss   : {log['avg_critic_loss'][0]:.3f} → {log['avg_critic_loss'][-1]:.3f}")
    print(f"  Entropy      : {log['avg_entropy'][0]:.4f} → {log['avg_entropy'][-1]:.4f}")
print("=" * 65)

# ─── CSV + Plots ─────────────────────────────────────────────────────────────

pd.DataFrame({'episode': log['episode'],
              'reward_raw':  log['reward_raw'],
              'reward_norm': log['reward_norm'],
              }).to_csv('training_log_mappo_episodes.csv', index=False)

if log['update_ep']:
    pd.DataFrame({'episode':         log['update_ep'],
                  'avg_actor_loss':  log['avg_actor_loss'],
                  'avg_critic_loss': log['avg_critic_loss'],
                  'avg_entropy':     log['avg_entropy'],
                  'avg_kl':          log['avg_kl'],
                  }).to_csv('training_log_mappo_updates.csv', index=False)

print("  ✓ CSV files saved.")

def smooth(data, w=50):
    if len(data) < w: return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('MAPPO Training — Robot Shop MARL (CTDE)', fontsize=14, fontweight='bold')

eps   = np.array(log['episode'])
upd_e = np.array(log['update_ep']) if log['update_ep'] else np.array([])

ax = axes[0, 0]
ax.plot(eps, log['reward_raw'], alpha=0.15, color='steelblue', lw=0.8)
if len(log['reward_raw']) >= 50:
    ax.plot(eps[49:], smooth(log['reward_raw']), color='steelblue', lw=2, label='MA-50')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_title('Episode Reward (Raw)')
ax.set_xlabel('Episode')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
if len(upd_e):
    ax.plot(upd_e, log['avg_critic_loss'], alpha=0.3, color='crimson', lw=0.8)
    if len(log['avg_critic_loss']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_critic_loss'], 20), color='crimson', lw=2)
ax.set_title('Critic Loss (Global State → better estimate)')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
if len(upd_e):
    ax.plot(upd_e, log['avg_actor_loss'], alpha=0.3, color='darkorange', lw=0.8)
    if len(log['avg_actor_loss']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_actor_loss'], 20), color='darkorange', lw=2)
ax.set_title('Actor Loss')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if len(upd_e):
    ax.plot(upd_e, log['avg_entropy'], alpha=0.3, color='seagreen', lw=0.8)
    if len(log['avg_entropy']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_entropy'], 20), color='seagreen', lw=2)
ax.axhline(np.log(3), color='gray', ls='--', alpha=0.7,
           label=f'Uniform {np.log(3):.3f}')
ax.set_title('Policy Entropy')
ax.set_xlabel('Episode')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_plot_mappo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ training_plot_mappo.png saved.")

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
}, os.path.join(CHECKPOINT_DIR, 'mappo_final.pth'))

print(f"  ✓ Final model: {CHECKPOINT_DIR}/mappo_final.pth")
print("\nDone!")
