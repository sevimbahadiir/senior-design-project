

import os, random, time
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from marl_env import MarlEnvironment, SERVICES
from qmix_agent import QMIXSystem, WARMUP_STEPS

N_EPISODES     = 3000
SAVE_INTERVAL  = 500
LOG_INTERVAL   = 50
SEED           = 42
CHECKPOINT_DIR = 'checkpoints_qmix'
REWARD_NORM_EPS = 1e-8

class RunningNormalizer:
    def __init__(self):
        self.mean = 0.0; self.var = 1.0; self.count = 0
    def update(self, v):
        self.count += 1
        d = v - self.mean; self.mean += d / self.count
        self.var += (v - self.mean) * d / self.count - self.var / self.count
    def normalize(self, v):
        return (v - self.mean) / (np.sqrt(max(self.var, REWARD_NORM_EPS)) + REWARD_NORM_EPS)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env        = MarlEnvironment(seed=SEED)
qmix       = QMIXSystem(env)
normalizer = RunningNormalizer()

print("=" * 65)
print("QMIX Training")
print("=" * 65)
counts = qmix.param_count()
print(f"  Local Q total   : {counts['local_q_total']:,}")
print(f"  Mixing network  : {counts['mixing_network']:,}")
print(f"  Update/step     : every step (after warmup)")
print(f"  Reward norm     : RunningNormalizer")
print("=" * 65)

df_sar     = pd.read_csv('rl_dataset_sar_normalized.csv')
state_cols = [f'{s}_{m}' for s in SERVICES
              for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
initial_states = df_sar[state_cols].values.astype(np.float32)
print(f"\n  {len(initial_states)} initial states loaded.")

print(f"  Warmup: {WARMUP_STEPS} steps...", end='', flush=True)
obs = env.reset(initial_state=initial_states[0])
for _ in range(WARMUP_STEPS):
    gs      = env.global_state.copy()
    actions = qmix.select_actions(obs, training=True)
    nobs, rewards, done, info = env.step(actions)
    ngs  = env.global_state.copy()
    raw_r = info['global_reward']
    normalizer.update(raw_r)
    norm_r = normalizer.normalize(raw_r)
    qmix.buffer.push(obs, actions, norm_r, nobs, gs, ngs, done)
    qmix.step()
    obs = nobs if not done else env.reset(initial_state=initial_states[random.randint(0, len(initial_states)-1)])
print(" ✓\n")

log         = defaultdict(list)
train_start = time.time()
global_step = 0
print("Starting training...\n")

for episode in range(1, N_EPISODES + 1):
    idx  = random.randint(0, len(initial_states) - 1)
    obs  = env.reset(initial_state=initial_states[idx])
    ep_raw = 0.0; ep_loss = []

    while True:
        gs      = env.global_state.copy()
        actions = qmix.select_actions(obs, training=True)
        nobs, rewards, done, info = env.step(actions)
        ngs      = env.global_state.copy()
        global_step += 1

        raw_r = info['global_reward']
        normalizer.update(raw_r)
        norm_r = normalizer.normalize(raw_r)

        qmix.buffer.push(obs, actions, norm_r, nobs, gs, ngs, done)
        qmix.step()

        loss = qmix.update()
        if loss is not None:
            ep_loss.append(loss)

        ep_raw += raw_r
        obs     = nobs
        if done:
            break

    log['episode'].append(episode)
    log['reward_raw'].append(ep_raw)
    log['epsilon'].append(qmix.eps_sched.value())
    log['loss'].append(np.mean(ep_loss) if ep_loss else 0.0)

    if episode % LOG_INTERVAL == 0 or episode == 1:
        elapsed = time.time() - train_start
        eta     = (N_EPISODES - episode) / (episode / elapsed) / 60
        w       = min(LOG_INTERVAL, episode)
        avg_r   = np.mean(log['reward_raw'][-w:])
        avg_l   = np.mean(log['loss'][-w:])
        print(f"  Ep {episode:>5}/{N_EPISODES} | "
              f"Reward: {avg_r:>8.3f} | "
              f"Loss: {avg_l:>8.4f} | "
              f"ε: {qmix.eps_sched.value():.3f} | "
              f"Steps: {global_step:>6} | "
              f"ETA: {eta:.0f}min")

    if episode % SAVE_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'qmix_{episode}.pth')
        qmix.save(path)
        torch.save({'episode': episode, 'log': dict(log),
                    'normalizer': vars(normalizer)}, path + '.meta')
        print(f"\n  ✓ Checkpoint: {path}\n")

elapsed  = time.time() - train_start
first100 = np.mean(log['reward_raw'][:100])
last100  = np.mean(log['reward_raw'][-100:])
print("\n" + "=" * 65)
print("QMIX training complete.")
print(f"  Duration     : {elapsed/60:.1f} min")
print(f"  Global steps : {global_step:,}")
print(f"  First 100 ep : {first100:.3f}  →  Last 100 ep: {last100:.3f}  ({last100-first100:+.3f})")
print(f"  Final ε      : {qmix.eps_sched.value():.4f}")
print("=" * 65)

pd.DataFrame(log).to_csv('training_log_qmix.csv', index=False)
print("  ✓ training_log_qmix.csv")

def smooth(d, w=50):
    return np.convolve(d, np.ones(w)/w, 'valid') if len(d) >= w else np.array(d)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('QMIX Training — Robot Shop MARL', fontsize=13, fontweight='bold')
eps_arr = np.array(log['episode'])

ax = axes[0]
ax.plot(eps_arr, log['reward_raw'], alpha=0.15, color='teal', lw=0.8)
if len(log['reward_raw']) >= 50:
    ax.plot(eps_arr[49:], smooth(log['reward_raw']), color='teal', lw=2)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_title('Episode Reward'); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(eps_arr, log['loss'], alpha=0.15, color='crimson', lw=0.8)
if len(log['loss']) >= 50:
    ax.plot(eps_arr[49:], smooth(log['loss']), color='crimson', lw=2)
ax.set_title('QMIX Loss (Bellman)'); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(eps_arr, log['epsilon'], color='steelblue', lw=2)
ax.set_title('Epsilon'); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_plot_qmix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ training_plot_qmix.png")

qmix.save(os.path.join(CHECKPOINT_DIR, 'qmix_final.pth'))
print(f"  ✓ Final model: {CHECKPOINT_DIR}/qmix_final.pth")
print("\nDone!")
