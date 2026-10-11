import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Callable
from collections import defaultdict

from marl_env import MarlEnvironment, SERVICES, METRICS, SERVICE_IDX

N_EVAL_EPISODES = 200
SEED            = 42
TEST_SPLIT      = 0.2


def load_test_states(path='rl_dataset_sar_normalized.csv'):
    df         = pd.read_csv(path)
    state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS if f'{s}_{m}' in df.columns]
    states     = df[state_cols].values.astype(np.float32)
    split      = int(len(states) * (1 - TEST_SPLIT))
    return states[split:]


def extract_metrics(global_state, reward_cfg):
    result       = {}
    slo_violated = False
    for svc in SERVICES:
        start   = SERVICE_IDX[svc]
        pods    = global_state[start + 0]
        latency = global_state[start + 4]
        result[f'{svc}_latency'] = float(latency)
        result[f'{svc}_pods']    = float(pods)
        if reward_cfg[svc].get('in_reward', True) and latency > 0.5:
            slo_violated = True
    result['slo_violation'] = 1.0 if slo_violated else 0.0
    return result


def evaluate_policy(policy_fn, env, test_states, reward_cfg,
                    n_episodes=N_EVAL_EPISODES, seed=SEED):
    rng     = np.random.default_rng(seed)
    results = {'episode_rewards': [], 'step_metrics': []}

    for ep in range(n_episodes):
        idx           = rng.integers(0, len(test_states))
        obs           = env.reset(initial_state=test_states[idx])
        ep_reward     = 0.0
        ep_step_data  = []

        while True:
            actions              = policy_fn(obs)
            obs, rewards, done, info = env.step(actions)
            ep_reward           += info['global_reward']
            step_m               = extract_metrics(info['global_state'], reward_cfg)
            step_m['reward']     = info['global_reward']
            ep_step_data.append(step_m)
            if done:
                break

        results['episode_rewards'].append(ep_reward)
        results['step_metrics'].extend(ep_step_data)

    return results


def summarize(results, policy_name):
    rewards = results['episode_rewards']
    steps   = results['step_metrics']
    summary = {
        'policy':          policy_name,
        'mean_reward':     float(np.mean(rewards)),
        'std_reward':      float(np.std(rewards)),
        'median_reward':   float(np.median(rewards)),
        'slo_violation_%': float(np.mean([s['slo_violation'] for s in steps]) * 100),
    }
    for svc in SERVICES:
        summary[f'{svc}_mean_latency'] = float(np.mean([s[f'{svc}_latency'] for s in steps]))
        summary[f'{svc}_mean_pods']    = float(np.mean([s[f'{svc}_pods']    for s in steps]))
    return summary


# ─── Policies ────────────────────────────────────────────────────────────────

def make_random_policy():
    rng = np.random.default_rng(SEED)
    def policy(obs): return {svc: int(rng.integers(0, 3)) for svc in SERVICES}
    return policy


def make_rule_based_policy():
    def policy(obs):
        actions = {}
        for svc in SERVICES:
            lat = float(obs[svc][4])
            actions[svc] = 2 if lat > 0.6 else (0 if lat < 0.2 else 1)
        return actions
    return policy


def make_ippo_policy(checkpoint_path, env):
    from ippo_agent import build_agent_pool
    agents = build_agent_pool(env)
    ckpt   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for svc in SERVICES:
        agents[svc].actor.load_state_dict(ckpt['agents'][svc]['actor'])
        agents[svc].actor.eval()
    def policy(obs):
        actions = {}
        for svc in SERVICES:
            obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(agents[svc].actor(obs_t).argmax(dim=-1).item())
            actions[svc] = action
        return actions
    return policy


def make_mappo_policy(checkpoint_path, env):
    from mappo_agent import build_mappo_pool
    agents = build_mappo_pool(env)
    ckpt   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for svc in SERVICES:
        agents[svc].actor.load_state_dict(ckpt['agents'][svc]['actor'])
        agents[svc].actor.eval()
    def policy(obs):
        actions = {}
        for svc in SERVICES:
            obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(agents[svc].actor(obs_t).argmax(dim=-1).item())
            actions[svc] = action
        return actions
    return policy


def make_dqn_policy(checkpoint_path, env):
    from dqn_agent import build_dqn_pool
    agents = build_dqn_pool(env)
    ckpt   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for svc in SERVICES:
        agents[svc].q_net.load_state_dict(ckpt['agents'][svc]['q_net'])
        agents[svc].q_net.eval()
    def policy(obs):
        actions = {}
        for svc in SERVICES:
            obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(agents[svc].q_net(obs_t).argmax(dim=-1).item())
            actions[svc] = action
        return actions
    return policy


def make_a2c_policy(checkpoint_path, env):
    from a2c_agent import build_a2c_pool
    agents = build_a2c_pool(env)
    ckpt   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for svc in SERVICES:
        agents[svc].actor.load_state_dict(ckpt['agents'][svc]['actor'])
        agents[svc].actor.eval()
    def policy(obs):
        actions = {}
        for svc in SERVICES:
            obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(agents[svc].actor(obs_t).argmax(dim=-1).item())
            actions[svc] = action
        return actions
    return policy

def make_qmix_policy(checkpoint_path, env):
    from qmix_agent import QMIXSystem
    qmix = QMIXSystem(env)
    qmix.load(checkpoint_path)
    for svc in SERVICES:
        qmix.q_nets[svc].eval()
    def policy(obs):
        return qmix.select_actions(obs, training=False)
    return policy

# ─── Main Evaluation ─────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("MARL Final Evaluation")
    print("=" * 65)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env         = MarlEnvironment(seed=SEED)
    test_states = load_test_states()
    reward_cfg  = env.reward_cfg

    print(f"\n  Test states  : {len(test_states)}")
    print(f"  Episodes     : {N_EVAL_EPISODES} / policy")
    print(f"  Policies     : Random, Rule-based, IPPO, MAPPO, DQN, A2C, QMIX\n")

    policies = {
        'Random':     make_random_policy(),
        'Rule-based': make_rule_based_policy(),
        'IPPO':       make_ippo_policy('checkpoints_v2/ippo_v2_final.pth',      env),
        'MAPPO':      make_mappo_policy('checkpoints_mappo/mappo_final.pth',     env),
        'DQN':        make_dqn_policy('checkpoints_dqn/dqn_final.pth',          env),
        'A2C':        make_a2c_policy('checkpoints_a2c/a2c_final.pth',          env),
        'QMIX': make_qmix_policy('checkpoints_qmix/qmix_final.pth', env),
    }

    all_results   = {}
    all_summaries = []

    for name, policy_fn in policies.items():
        print(f"  [{name}] evaluating...", end='', flush=True)
        results = evaluate_policy(policy_fn, env, test_states, reward_cfg)
        summary = summarize(results, name)
        all_results[name]  = results
        all_summaries.append(summary)
        print(f" ✓  reward: {summary['mean_reward']:+.3f} ± {summary['std_reward']:.3f}")

    # ── Results Table ──
    print("\n" + "=" * 65)
    print("FINAL RESULTS TABLE")
    print("=" * 65)
    print(f"\n  {'Policy':<14} {'Mean Reward':>12} {'Std':>8} "
          f"{'Median':>8} {'SLO Viol.%':>12}")
    print(f"  {'-'*58}")
    for s in sorted(all_summaries, key=lambda x: -x['mean_reward']):
        print(f"  {s['policy']:<14} {s['mean_reward']:>+12.3f} "
              f"{s['std_reward']:>8.3f} {s['median_reward']:>+8.3f} "
              f"{s['slo_violation_%']:>11.1f}%")

    # ── Latency Table ──
    print(f"\n  Mean Latency (normalized, lower=better):")
    names = [s['policy'] for s in all_summaries]
    print(f"  {'Service':<12}" + "".join(f"  {n:>12}" for n in names))
    print(f"  {'-'*75}")
    for svc in SERVICES:
        row = f"  {svc:<12}"
        for s in all_summaries:
            row += f"  {s[f'{svc}_mean_latency']:>12.4f}"
        print(row)

    # ── Pod Table ──
    print(f"\n  Mean Pod Count (normalized):")
    print(f"  {'Service':<12}" + "".join(f"  {n:>12}" for n in names))
    print(f"  {'-'*75}")
    for svc in SERVICES:
        row = f"  {svc:<12}"
        for s in all_summaries:
            row += f"  {s[f'{svc}_mean_pods']:>12.4f}"
        print(row)

    pd.DataFrame(all_summaries).to_csv('results/evaluation_summary_final.csv', index=False)
    print(f"\n  ✓ evaluation_summary_final.csv saved.")

    # ── Plots ──
    colors = {'Random':'lightcoral','Rule-based':'steelblue','IPPO':'seagreen',
              'MAPPO':'gold','DQN':'darkorange','A2C':'mediumpurple','QMIX': 'teal'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MARL Final Comparison — Robot Shop', fontsize=14, fontweight='bold')

    rewards = [s['mean_reward'] for s in all_summaries]
    stds    = [s['std_reward']  for s in all_summaries]
    slos    = [s['slo_violation_%'] for s in all_summaries]
    clrs    = [colors.get(n, 'gray') for n in names]

    ax = axes[0]
    bars = ax.bar(names, rewards, color=clrs, alpha=0.85,
                  yerr=stds, capsize=5, edgecolor='black', lw=0.8)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_title('Mean Episode Reward\n(higher = better)')
    ax.set_ylabel('Reward')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.5 if val >= 0 else -2),
                f'{val:+.1f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(names, slos, color=clrs, alpha=0.85,
                  edgecolor='black', lw=0.8)
    ax.set_title('SLO Violation Rate\n(lower = better)')
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, slos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax = axes[2]
    data_box = [all_results[n]['episode_rewards'] for n in names]
    bp = ax.boxplot(data_box, tick_labels=names, patch_artist=True,
                    medianprops={'color':'black','linewidth':2})
    for patch, clr in zip(bp['boxes'], clrs):
        patch.set_facecolor(clr); patch.set_alpha(0.7)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_title('Reward Distribution\n(box plot)')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/evaluation_plot_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ evaluation_plot_final.png saved.")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
