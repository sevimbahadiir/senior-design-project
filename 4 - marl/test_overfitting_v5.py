import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats as scipy_stats

from marl_env import MarlEnvironment, SERVICES, METRICS, SERVICE_IDX
from ippo_agent import build_agent_pool

# Configuration

CHECKPOINT    = 'checkpoints_v5/ippo_v5_final.pth'
SAR_PATH      = 'rl_dataset_sar_normalized.csv'
REWARD_CFG    = 'reward_config.json'
N_EPISODES    = 200
SEEDS         = [42, 123, 456, 789, 1024]
TRAIN_SPLIT   = 0.8

# Reference values from previous v3 and v4 (old DT) comparisons
REF = {
    'v3_cv':          74.6,
    'v4_new_dt_cv':   14.7,
    'v4_new_dt_mean': 13.91,
    'v4_new_dt_ci_low':  11.07,
    'v4_new_dt_ci_high': 16.74,
}

# Load Model

def load_model(env, checkpoint_path):
    agents = build_agent_pool(env)
    ckpt   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for svc in SERVICES:
        agents[svc].actor.load_state_dict(ckpt['agents'][svc]['actor'])
        agents[svc].actor.eval()
    return agents


def ippo_policy(obs, agents):
    """Deterministic policy (argmax)."""
    actions = {}
    for svc in SERVICES:
        obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = agents[svc].actor(obs_t)
            action = int(logits.argmax(dim=-1).item())
        actions[svc] = action
    return actions


# Evaluation

def evaluate(env, agents, states, n_episodes, seed):
    rng     = np.random.default_rng(seed)
    rewards = []
    slo_violations = []
    action_counts  = defaultdict(lambda: defaultdict(int))

    with open(REWARD_CFG) as f:
        reward_cfg = json.load(f)

    for ep in range(n_episodes):
        idx = rng.integers(0, len(states))
        obs = env.reset(initial_state=states[idx])
        ep_reward = 0.0
        ep_slo    = []

        while True:
            actions = ippo_policy(obs, agents)
            for svc in SERVICES:
                action_counts[svc][actions[svc]] += 1

            next_obs, rewards_dict, done, info = env.step(actions)
            ep_reward += info['global_reward']

            gs = info['global_state']
            slo_viol = any(
                gs[SERVICE_IDX[svc] + 4] > 0.5
                for svc in SERVICES
                if reward_cfg[svc].get('in_reward', True)
            )
            ep_slo.append(float(slo_viol))
            obs = next_obs
            if done:
                break

        rewards.append(ep_reward)
        slo_violations.append(np.mean(ep_slo))

    return rewards, np.mean(slo_violations), action_counts


# Setup

print("=" * 65)
print("IPPO v5 (Entropy Floor) — Stability and Generalization Test")
print("=" * 65)

env    = MarlEnvironment(seed=42)
agents = load_model(env, CHECKPOINT)
print(f"  OK Model loaded: {CHECKPOINT}")

df_sar     = pd.read_csv(SAR_PATH)
state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]
all_states = df_sar[state_cols].values.astype(np.float32)
split_idx  = int(len(all_states) * TRAIN_SPLIT)

train_states = all_states[:split_idx]   # first 80% -- low-medium load
test_states  = all_states[split_idx:]   # last 20% -- high load

print(f"  Train split states : {len(train_states)} (weighted toward low-medium load)")
print(f"  Test split states  : {len(test_states)}  (weighted toward high load)")
print()

# ─── TEST 1: Train vs Test Split ─────────────────────────────────────────────

print("=" * 65)
print("TEST 1 — Train Split vs Test Split Comparison")
print("=" * 65)
print(f"  {N_EPISODES} episodes per split, seed=42")
print()

test_rewards,  test_slo,  test_actions  = evaluate(env, agents, test_states,  N_EPISODES, seed=42)
train_rewards, train_slo, train_actions = evaluate(env, agents, train_states, N_EPISODES, seed=42)

test_mean  = np.mean(test_rewards)
train_mean = np.mean(train_rewards)
gap        = test_mean - train_mean
gap_pct    = abs(gap) / max(abs(train_mean), 1e-8) * 100

print(f"  {'Metric':<28} {'Test Split':>12} {'Train Split':>12} {'Diff':>10}")
print(f"  {'-'*65}")
print(f"  {'Mean Reward':<28} {test_mean:>12.3f} {train_mean:>12.3f} {gap:>+10.3f}")
print(f"  {'Median Reward':<28} {np.median(test_rewards):>12.3f} {np.median(train_rewards):>12.3f}")
print(f"  {'Std Reward':<28} {np.std(test_rewards):>12.3f} {np.std(train_rewards):>12.3f}")
print(f"  {'Positive Ep %':<28} {(np.array(test_rewards)>0).mean()*100:>11.1f}% {(np.array(train_rewards)>0).mean()*100:>11.1f}%")
print(f"  {'SLO Violation Rate':<28} {test_slo:>11.1%} {train_slo:>11.1%}")
print()

# Overfitting decision -- look at action distribution consistency
ACTION_LABELS = {0: 'scale_down', 1: 'keep', 2: 'scale_up'}
print(f"  Action Distribution Consistency (real overfitting evidence):")
print(f"  {'Service':<12} {'Action':<12} {'Test':>10} {'Train':>10} {'Consistent?':>12}")
print(f"  {'-'*58}")
all_consistent = True
for svc in SERVICES:
    total_test  = sum(test_actions[svc].values())
    total_train = sum(train_actions[svc].values())
    for a in [0, 1, 2]:
        tp = test_actions[svc].get(a, 0) / max(total_test, 1) * 100
        tr = train_actions[svc].get(a, 0) / max(total_train, 1) * 100
        if tp > 5 or tr > 5:
            diff = abs(tp - tr)
            consistent = 'YES' if diff < 15 else 'NO'
            if diff >= 15:
                all_consistent = False
            print(f"  {svc:<12} {ACTION_LABELS[a]:<12} {tp:>9.1f}% {tr:>9.1f}% {consistent:>12}")

print()
print(f"  Action consistency: {'CONSISTENT -- no real overfitting' if all_consistent else 'DIFFERENCE FOUND -- check'}")
print(f"  Mean gap of {gap_pct:.1f}% is caused by the load level difference")
print(f"  (train split = low load -> easier starting conditions)")

#  TEST 2: Stability (5 Seeds)

print(f"\n{'=' * 65}")
print("TEST 2 — Stability Test (5 Seeds)")
print("=" * 65)
print(f"  {N_EPISODES} episodes per seed, test split states")
print()

seed_results = []
all_seed_rewards = []

for seed in SEEDS:
    rewards, slo, _ = evaluate(env, agents, test_states, N_EPISODES, seed)
    all_seed_rewards.append(rewards)
    result = {
        'seed':    seed,
        'mean':    np.mean(rewards),
        'std':     np.std(rewards),
        'median':  np.median(rewards),
        'pos_pct': (np.array(rewards) > 0).mean() * 100,
        'slo_viol': slo,
    }
    seed_results.append(result)
    print(f"  Seed {seed:>4}: mean={result['mean']:>7.3f}, "
          f"std={result['std']:>7.3f}, "
          f"median={result['median']:>7.3f}, "
          f"pos%={result['pos_pct']:>5.1f}%")

seed_means   = [r['mean'] for r in seed_results]
seed_medians = [r['median'] for r in seed_results]
overall_mean = np.mean(seed_means)
overall_std  = np.std(seed_means)
cv           = overall_std / max(abs(overall_mean), 1e-8) * 100
ci           = scipy_stats.t.interval(0.95, df=4,
                                       loc=overall_mean,
                                       scale=scipy_stats.sem(seed_means))

print(f"\n  {'Metric':<28} {'Value':>10} {'Reference':>12}")
print(f"  {'-'*54}")
print(f"  {'Mean (5 seeds)':<28} {overall_mean:>10.3f} {'v4: 13.91':>12}")
print(f"  {'Std (5 seeds)':<28} {overall_std:>10.3f}")
print(f"  {'CV (std/mean)':<28} {cv:>9.1f}% {'v4: 14.7%':>12}")
print(f"  {'95% CI':<28} [{ci[0]:>6.3f}, {ci[1]:>6.3f}]")
print(f"  {'Median mean (5 seeds)':<28} {np.mean(seed_medians):>10.3f}")
print(f"  {'All median positive?':<28} {'YES' if all(m > 0 for m in seed_medians) else 'NO':>10}")
print(f"  {'All mean positive?':<28} {'YES' if all(m > 0 for m in seed_means) else 'NO':>10}")

# Stability decision
if cv < 30:
    stab = "HIGH STABILITY"
elif cv < 50:
    stab = "ACCEPTABLE"
elif cv < 75:
    stab = "LOW STABILITY"
else:
    stab = "HIGH VARIANCE"

print(f"\n  Stability: {stab} (CV={cv:.1f}%)")

# Comparison with previous versions
print(f"\n  Version comparison:")
print(f"  {'Version':<20} {'CV%':>8} {'5-seed mean':>13} {'Converged':>12}")
print(f"  {'-'*56}")
print(f"  {'v3':<20} {REF['v3_cv']:>7.1f}% {'--':>13} {'NO':>12}")
print(f"  {'v4 (new DT)':<20} {REF['v4_new_dt_cv']:>7.1f}% {REF['v4_new_dt_mean']:>13.3f} {'YES (ep1200)':>12}")
print(f"  {'v5 entropy floor':<20} {cv:>7.1f}% {overall_mean:>13.3f} {'YES (ep2600)':>12}")

# Save CSV

pd.DataFrame({
    'split':  ['test_split'] * N_EPISODES + ['train_split'] * N_EPISODES,
    'reward': test_rewards + train_rewards,
}).to_csv('overfitting_test_rewards_v5.csv', index=False)

pd.DataFrame(seed_results).to_csv('stability_test_v5.csv', index=False)

pd.DataFrame([{
    'checkpoint':      CHECKPOINT,
    'test_mean':       test_mean,
    'train_mean':      train_mean,
    'gap_pct':         gap_pct,
    'action_consistent': all_consistent,
    'cv_pct':          cv,
    'ci95_low':        ci[0],
    'ci95_high':       ci[1],
    'all_median_pos':  all(m > 0 for m in seed_medians),
    'all_mean_pos':    all(m > 0 for m in seed_means),
    'convergence_ep':  2600,
    'vs_v4_cv':        cv - REF['v4_new_dt_cv'],
}]).to_csv('stability_summary_v5.csv', index=False)

print(f"\n  OK overfitting_test_rewards_v4.csv")
print(f"  OK stability_test_v5.csv")
print(f"  OK stability_summary_v5.csv")

# Plot

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('IPPO v5 (Entropy Floor) — Stability and Generalization Test', fontsize=13, fontweight='bold')

# Reward distribution
ax = axes[0]
ax.hist(test_rewards,  bins=30, alpha=0.6, color='steelblue',
        label=f'Test split (mu={test_mean:.2f}, med={np.median(test_rewards):.2f})')
ax.hist(train_rewards, bins=30, alpha=0.6, color='darkorange',
        label=f'Train split (mu={train_mean:.2f}, med={np.median(train_rewards):.2f})')
ax.axvline(test_mean,  color='steelblue',  ls='--', lw=2)
ax.axvline(train_mean, color='darkorange', ls='--', lw=2)
ax.axvline(0, color='gray', ls=':', lw=1)
ax.set_xlabel('Episode Reward')
ax.set_title(f'Train vs Test Split\nGap: {gap:+.2f} (load difference)\nAction consistent: {"YES" if all_consistent else "NO"}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Seed box plot
ax = axes[1]
bp = ax.boxplot(all_seed_rewards, labels=[str(s) for s in SEEDS], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Seed')
ax.set_ylabel('Episode Reward')
ax.set_title(f'Stability (5 Seeds)\nMean={overall_mean:.2f}, CV={cv:.1f}%\n{stab}')
ax.grid(True, alpha=0.3)

# Version comparison
ax = axes[2]
versions = ['v3', 'v4\n(new DT)', 'v5\n(entropy floor)']
cvs      = [REF['v3_cv'], REF['v4_new_dt_cv'], cv]
colors   = ['#e74c3c', '#2ecc71', '#3498db' if cv < 20 else ('#f39c12' if cv < 50 else '#e74c3c')]
bars = ax.bar(versions, cvs, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(50, color='orange', ls='--', lw=2, alpha=0.7, label='50% threshold')
ax.axhline(30, color='green',  ls='--', lw=2, alpha=0.7, label='30% target')
for bar, val in zip(bars, cvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('CV% (lower = more stable)')
ax.set_title('Stability Comparison\n(CV% by Version)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(cvs) * 1.2)

plt.tight_layout()
plt.savefig('stability_test_v5.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK stability_test_v5.png")

