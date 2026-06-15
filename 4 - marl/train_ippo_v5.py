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
from ippo_agent import build_agent_pool

# Hyperparameters

N_EPISODES          = 10000
EPISODES_PER_UPDATE = 5
SAVE_INTERVAL       = 500
LOG_INTERVAL        = 100
SEED                = 42
CHECKPOINT_DIR      = 'checkpoints_v5'

# Entropy decay 
ENTROPY_START = 0.05
ENTROPY_END   = 0.01   

CONV_WINDOW    = 800  
CONV_THRESHOLD = 0.05  


# Entropy Decay 

def get_entropy_coef(episode: int) -> float:
    """
    Linear entropy decay.
    Episode 1     -> 0.05  (high exploration)
    Episode 10000 -> 0.005 (low exploration, lock into policy)
    """
    progress = (episode - 1) / max(N_EPISODES - 1, 1)
    return ENTROPY_START + (ENTROPY_END - ENTROPY_START) * progress


# Convergence Detection 

def check_convergence(reward_history: list, episode: int) -> tuple:

    if episode < 2 * CONV_WINDOW:
        return False, None

    recent   = np.mean(reward_history[-CONV_WINDOW:])
    previous = np.mean(reward_history[-2*CONV_WINDOW:-CONV_WINDOW])

    if abs(previous) < 1e-8:
        return False, None

    relative_change = abs(recent - previous) / abs(previous)

    if relative_change < CONV_THRESHOLD:
        plateau_ep = episode - CONV_WINDOW
        return True, plateau_ep

    return False, None


# Setup

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env    = MarlEnvironment(seed=SEED)
agents = build_agent_pool(env)

print("=" * 65)
print("IPPO Training v5 — Entropy Floor (Collapsed Policy Fix)")
print("=" * 65)
print(f"  N_EPISODES      : {N_EPISODES}")
print(f"  Episodes/update : {EPISODES_PER_UPDATE} ep = {EPISODES_PER_UPDATE*10} transitions")
print(f"  Entropy decay   : {ENTROPY_START} -> {ENTROPY_END}  (v4: 0.005)")
print(f"  Reward          : RAW tanh reward  (v3: was normalized)")
print(f"  Expected range  : [-8, +3]         (natural bound from tanh)")
print(f"  Entropy floor   : 0.3  (torch.clamp in ippo_agent.py)\n"
    f"  Convergence     : window={CONV_WINDOW} ep, threshold={CONV_THRESHOLD*100:.0f}%")
print(f"                    (v4: window=500, threshold=5%)")
print(f"  LR actor/critic : 1e-4 / 5e-4")
print(f"  K_epochs        : 8")
print("=" * 65)

# SAR Dataset

df_sar     = pd.read_csv('rl_dataset_sar_normalized.csv')
state_cols = [
    f'{s}_{m}' for s in SERVICES
    for m in ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
]
initial_states = df_sar[state_cols].values.astype(np.float32)
print(f"\n  {len(initial_states)} initial states ready.\n")

# Log

log = defaultdict(list)

# Training Loop

print("Training starting...\n")
train_start     = time.time()
update_count    = 0
converged       = False
plateau_episode = None

for episode in range(1, N_EPISODES + 1):

    # 1. Reset -- pick a random initial state
    idx = random.randint(0, len(initial_states) - 1)
    obs = env.reset(initial_state=initial_states[idx])

    ep_reward    = 0.0
    entropy_coef = get_entropy_coef(episode)

    # 2. Episode -- 10 steps
    while True:
        actions = {}
        for svc in SERVICES:
            action, log_prob, value = agents[svc].select_action(obs[svc])
            actions[svc] = action
            agents[svc].buffer.push(
                obs=obs[svc], action=action, log_prob=log_prob,
                reward=0.0, done=False, value=value,
            )

        next_obs, rewards, done, info = env.step(actions)

        raw_r = info['global_reward']

        for svc in SERVICES:
            agents[svc].buffer.rewards[-1] = raw_r   # raw reward directly
            agents[svc].buffer.dones[-1]   = done

        ep_reward += raw_r
        obs        = next_obs

        if done:
            break

    log['episode'].append(episode)
    log['reward_raw'].append(ep_reward)
    log['entropy_coef'].append(entropy_coef)

    # 3. Update
    if episode % EPISODES_PER_UPDATE == 0:
        update_count += 1
        ms = {}
        for svc in SERVICES:
            ms[svc] = agents[svc].update(
                last_obs=obs[svc],
                entropy_coef=entropy_coef,
            )
            agents[svc].buffer.clear()

        log['update_ep'].append(episode)
        log['avg_actor_loss'].append(np.mean([ms[s]['actor_loss']  for s in SERVICES]))
        log['avg_critic_loss'].append(np.mean([ms[s]['critic_loss'] for s in SERVICES]))
        log['avg_entropy'].append(np.mean([ms[s]['entropy']        for s in SERVICES]))
        log['avg_kl'].append(np.mean([ms[s]['approx_kl']           for s in SERVICES]))

    # 4. Convergence check
    if episode % 100 == 0 and not converged:
        converged, plateau_episode = check_convergence(log['reward_raw'], episode)
        if converged:
            print(f"\n  OK CONVERGENCE DETECTED!")
            print(f"    Plateau episode      : {plateau_episode}")
            print(f"    Last {CONV_WINDOW} ep avg     : {np.mean(log['reward_raw'][-CONV_WINDOW:]):.4f}")
            print(f"    Previous {CONV_WINDOW} ep avg : {np.mean(log['reward_raw'][-2*CONV_WINDOW:-CONV_WINDOW]):.4f}")
            print(f"    Training continues...\n")

    # 5. Log
    if episode % LOG_INTERVAL == 0 or episode == 1:
        elapsed  = time.time() - train_start
        eta      = (N_EPISODES - episode) / max(episode / elapsed, 1e-8) / 60
        w        = min(LOG_INTERVAL, episode)
        avg_r    = np.mean(log['reward_raw'][-w:])
        al = log['avg_actor_loss'][-1]  if log['avg_actor_loss']  else 0
        cl = log['avg_critic_loss'][-1] if log['avg_critic_loss'] else 0
        en = log['avg_entropy'][-1]     if log['avg_entropy']     else 0
        conv_str = f"CONVERGED(ep{plateau_episode})" if converged else "running"
        print(
            f"  Ep {episode:>5}/{N_EPISODES} | "
            f"Reward: {avg_r:>8.4f} | "
            f"Entropy_c: {entropy_coef:.4f} | "
            f"ActorL: {al:>7.4f} | "
            f"CriticL: {cl:>6.3f} | "
            f"Ent: {en:.4f} | "
            f"ETA: {eta:.0f}min | "
            f"{conv_str}"
        )

    # 6. Checkpoint
    if episode % SAVE_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'ippo_v5_{episode}.pth')
        torch.save({
            'episode':      episode,
            'converged':    converged,
            'plateau_ep':   plateau_episode,
            'entropy_coef': entropy_coef,
            'agents': {svc: {
                'actor':     agents[svc].actor.state_dict(),
                'critic':    agents[svc].critic.state_dict(),
                'actor_opt': agents[svc].actor_opt.state_dict(),
                'critic_opt':agents[svc].critic_opt.state_dict(),
            } for svc in SERVICES},
            'log': dict(log),
        }, path)
        print(f"\n  OK Checkpoint: {path}\n")

# End of Training

elapsed  = time.time() - train_start
first100 = np.mean(log['reward_raw'][:100])
last100  = np.mean(log['reward_raw'][-100:])

print("\n" + "=" * 65)
print("Training complete.")
print(f"  Duration       : {elapsed/60:.1f} min  |  Updates: {update_count}")
print(f"  First 100 ep   : {first100:.4f}")
print(f"  Last 100 ep    : {last100:.4f}  ({last100-first100:+.4f})")
print(f"  Expected range : [-8, +3] — actual: [{min(log['reward_raw']):.2f}, {max(log['reward_raw']):.2f}]")
print(f"  Convergence    : {'YES — ep ' + str(plateau_episode) if converged else 'NO'}")
if log['avg_critic_loss']:
    print(f"  CriticLoss     : {log['avg_critic_loss'][0]:.4f} -> {log['avg_critic_loss'][-1]:.4f}")
    print(f"  Entropy        : {log['avg_entropy'][0]:.4f} -> {log['avg_entropy'][-1]:.4f}")
print("=" * 65)

pd.DataFrame({
    'episode':      log['episode'],
    'reward_raw':   log['reward_raw'],
    'entropy_coef': log['entropy_coef'],
}).to_csv('training_log_v5_episodes.csv', index=False)

if log['update_ep']:
    pd.DataFrame({
        'episode':         log['update_ep'],
        'avg_actor_loss':  log['avg_actor_loss'],
        'avg_critic_loss': log['avg_critic_loss'],
        'avg_entropy':     log['avg_entropy'],
        'avg_kl':          log['avg_kl'],
    }).to_csv('training_log_v5_updates.csv', index=False)

print("  OK CSVs saved (v5).")

pd.DataFrame([{
    'version':         'v5',
    'total_episodes':  N_EPISODES,
    'reward_norm':     'NO — raw tanh reward',
    'entropy_floor':   0.3,
    'plateau_episode': plateau_episode if converged else N_EPISODES,
    'converged':       converged,
    'first100':        first100,
    'last100':         last100,
    'improvement':     last100 - first100,
    'reward_min':      min(log['reward_raw']),
    'reward_max':      max(log['reward_raw']),
    'entropy_start':   ENTROPY_START,
    'entropy_end':     ENTROPY_END,
    'conv_window':     CONV_WINDOW,
    'conv_threshold':  CONV_THRESHOLD,
}]).to_csv('convergence_v5.csv', index=False)
print("  OK convergence_v5.csv saved.")

# ─── Plots ────────────────────────────────────────────────────────────────────

def smooth(data, w=100):
    if len(data) < w:
        return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode='valid')

eps   = np.array(log['episode'])
upd_e = np.array(log['update_ep']) if log['update_ep'] else np.array([])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    'IPPO v5 — Entropy Floor (Collapsed Policy Fix)\n'
    'Entropy floor=0.3 | ENTROPY_END=0.01 | CONV_WINDOW=800',
    fontsize=13, fontweight='bold'
)

# Reward
ax = axes[0, 0]
ax.plot(eps, log['reward_raw'], alpha=0.15, color='steelblue', lw=0.5)
if len(log['reward_raw']) >= 100:
    ax.plot(eps[99:], smooth(log['reward_raw'], 100),
            color='steelblue', lw=2, label='Avg-100')
ax.axhline(0, color='gray', ls='--', alpha=0.5, label='Zero line')
ax.axhline(-2.43, color='orange', ls=':', alpha=0.7, label='SAR mean (-2.43)')
if converged and plateau_episode:
    ax.axvline(plateau_episode, color='green', ls='--', alpha=0.8,
               label=f'Convergence ep{plateau_episode}')
ax.set_title('Episode Reward (Raw Tanh)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Reward distribution -- for comparison with v3
ax = axes[0, 1]
ax.hist(log['reward_raw'], bins=60, color='steelblue', alpha=0.7, edgecolor='none')
ax.axvline(np.mean(log['reward_raw']), color='navy', ls='--', lw=2,
           label=f"Mean: {np.mean(log['reward_raw']):.3f}")
ax.axvline(0, color='gray', ls=':', lw=1)
ax.set_title('Reward Distribution\n(v3 had a -60/+45 range, v4 expected narrower)')
ax.set_xlabel('Episode Reward')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# Entropy
ax = axes[0, 2]
if len(upd_e):
    ax.plot(upd_e, log['avg_entropy'], alpha=0.3, color='seagreen', lw=0.8)
    if len(log['avg_entropy']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_entropy'], 20),
                color='seagreen', lw=2)
ax.axhline(np.log(3), color='gray', ls='--', alpha=0.7,
           label=f'Uniform {np.log(3):.3f}')
ax.set_title('Policy Entropy (gradual decrease expected)')
ax.set_xlabel('Episode')
ax.legend()
ax.grid(True, alpha=0.3)

# Critic Loss
ax = axes[1, 0]
if len(upd_e):
    ax.plot(upd_e, log['avg_critic_loss'], alpha=0.3, color='crimson', lw=0.8)
    if len(log['avg_critic_loss']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_critic_loss'], 20),
                color='crimson', lw=2)
ax.set_title('Critic Loss')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

# Actor Loss
ax = axes[1, 1]
if len(upd_e):
    ax.plot(upd_e, log['avg_actor_loss'], alpha=0.3, color='darkorange', lw=0.8)
    if len(log['avg_actor_loss']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_actor_loss'], 20),
                color='darkorange', lw=2)
ax.set_title('Actor Loss')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

# Zoom on last 2000 episodes
ax = axes[1, 2]
last_eps  = eps[-2000:]
last_rews = log['reward_raw'][-2000:]
ax.plot(last_eps, last_rews, alpha=0.15, color='steelblue', lw=0.5)
if len(last_rews) >= 100:
    ax.plot(last_eps[99:], smooth(last_rews, 100),
            color='steelblue', lw=2, label='Avg-100')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
if converged and plateau_episode and plateau_episode > last_eps[0]:
    ax.axvline(plateau_episode, color='green', ls='--', alpha=0.8)
ax.set_title('Last 2000 Episode Rewards (Zoom)')
ax.set_xlabel('Episode')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_plot_v5.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK training_plot_v5.png saved.")

# Final Model

torch.save({
    'episode':      N_EPISODES,
    'converged':    converged,
    'plateau_ep':   plateau_episode,
    'version':      'v5',
    'reward_norm':  False,
    'agents': {svc: {
        'actor':     agents[svc].actor.state_dict(),
        'critic':    agents[svc].critic.state_dict(),
        'actor_opt': agents[svc].actor_opt.state_dict(),
        'critic_opt':agents[svc].critic_opt.state_dict(),
    } for svc in SERVICES},
    'log': dict(log),
}, os.path.join(CHECKPOINT_DIR, 'ippo_v5_final.pth'))

print(f"  OK Final model: {CHECKPOINT_DIR}/ippo_v4_final.pth")
print("\nDone!")
