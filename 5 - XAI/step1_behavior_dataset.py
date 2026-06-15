import numpy as np
import pandas as pd
import torch
import json

from marl_env import MarlEnvironment, SERVICES, METRICS, SERVICE_IDX
from ippo_agent import build_agent_pool

IPPO_CHECKPOINT = 'checkpoints_v5/ippo_v5_final.pth'
DT_PATH         = 'digital_twin_best.pth'
REWARD_CFG_PATH = 'reward_config.json'
SCALER_PATH     = 'scaler.pkl'
SAR_PATH        = 'rl_dataset_sar_normalized.csv'
N_EPISODES      = 200
SEED            = 42
TEST_SPLIT      = 0.2
ACTION_LABELS   = {0: 'scale_down', 1: 'keep', 2: 'scale_up'}

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 65)
print("XAI Step 1: Behavior Dataset Collection")
print("=" * 65)

env    = MarlEnvironment(dt_path=DT_PATH, reward_cfg_path=REWARD_CFG_PATH,
                         scaler_path=SCALER_PATH, seed=SEED)
agents = build_agent_pool(env)
ckpt   = torch.load(IPPO_CHECKPOINT, map_location='cpu', weights_only=False)
for svc in SERVICES:
    agents[svc].actor.load_state_dict(ckpt['agents'][svc]['actor'])
    agents[svc].actor.eval()
print(f"  OK IPPO model loaded: {IPPO_CHECKPOINT}")

df_sar      = pd.read_csv(SAR_PATH)
state_cols  = [f'{s}_{m}' for s in SERVICES for m in METRICS]
states_all  = df_sar[state_cols].values.astype(np.float32)
split       = int(len(states_all) * (1 - TEST_SPLIT))
test_states = states_all[split:]
print(f"  OK Test states: {len(test_states)}")

#Episode loop

rng     = np.random.default_rng(SEED)
records = []

print(f"\n  Running {N_EPISODES} episodes x 10 steps...")

for ep in range(N_EPISODES):
    idx  = rng.integers(0, len(test_states))
    obs  = env.reset(initial_state=test_states[idx])
    step = 0

    while True:
        global_state = env.global_state.copy()

        for svc in SERVICES:
            obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = agents[svc].actor(obs_t)
                probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
                action = int(probs.argmax())

            record = {
                'episode': ep, 'step': step, 'service': svc,
                'action':  action,
                'action_label':   ACTION_LABELS[action],
                'prob_scale_down': float(probs[0]),
                'prob_keep':       float(probs[1]),
                'prob_scale_up':   float(probs[2]),
                'confidence':      float(probs.max()),
            }
            for i, val in enumerate(obs[svc]):
                record[f'obs_{i}'] = float(val)

            start = SERVICE_IDX[svc]
            for j, metric in enumerate(METRICS):
                record[f'gs_{metric}'] = float(global_state[start + j])

            records.append(record)

        actions_dict = {}
        for svc in SERVICES:
            svc_records = [r for r in records
                           if r['episode'] == ep
                           and r['step'] == step
                           and r['service'] == svc]
            actions_dict[svc] = svc_records[-1]['action']

        next_obs, rewards, done, info = env.step(actions_dict)

        for i, svc in enumerate(SERVICES):
            records[-(len(SERVICES) - i)]['reward'] = info['global_reward']

        obs  = next_obs
        step += 1
        if done:
            break

    if (ep + 1) % 50 == 0:
        print(f"    Episode {ep+1}/{N_EPISODES} completed")

df_all = pd.DataFrame(records)
print(f"\n  Total records: {len(df_all)}")

#  Action distribution 

print(f"\n  Action Distribution:")
print(f"  {'Service':<12} {'scale_down':>12} {'keep':>8} {'scale_up':>10} {'Dominant':>12}")
print(f"  {'-'*58}")
for svc in SERVICES:
    sdf   = df_all[df_all['service'] == svc]
    total = len(sdf)
    cnts  = sdf['action_label'].value_counts()
    sd    = cnts.get('scale_down', 0)
    k     = cnts.get('keep', 0)
    su    = cnts.get('scale_up', 0)
    dom   = cnts.index[0]
    print(f"  {svc:<12} {sd:>8}({sd/total*100:.0f}%) "
          f"{k:>4}({k/total*100:.0f}%) "
          f"{su:>6}({su/total*100:.0f}%)  {dom}")

print(f"\n  Average Decision Confidence:")
print(f"  {'Service':<12} {'Avg. Conf.':>12} {'Min':>8} {'Max':>8}")
print(f"  {'-'*44}")
for svc in SERVICES:
    conf = df_all[df_all['service'] == svc]['confidence']
    print(f"  {svc:<12} {conf.mean():>12.3f} {conf.min():>8.3f} {conf.max():>8.3f}")

#  Save: all services 

df_all.to_csv('xai_behavior_dataset.csv', index=False)
print(f"\n  OK xai_behavior_dataset.csv ({len(df_all)} rows)")

# Obs map

from marl_env import AGENT_NEIGHBORS

obs_map = {}
for svc in SERVICES:
    neighbors = AGENT_NEIGHBORS[svc]
    all_svcs  = [svc] + neighbors
    obs_map[svc] = {}
    for i, s in enumerate(all_svcs):
        for j, metric in enumerate(METRICS):
            col_idx = i * len(METRICS) + j
            obs_map[svc][f'obs_{col_idx}'] = {
                'service': s, 'metric': metric,
                'role': 'self' if s == svc else 'neighbor'
            }

with open('xai_obs_map.json', 'w') as f:
    json.dump(obs_map, f, indent=2)
print(f"  OK xai_obs_map.json")

# Save: per service

for svc in SERVICES:
    svc_df   = df_all[df_all['service'] == svc].copy()
    n_obs    = len(obs_map[svc])
    obs_cols = [f'obs_{i}' for i in range(n_obs)]
    out_cols = obs_cols + ['action', 'action_label',
                           'prob_scale_down', 'prob_keep',
                           'prob_scale_up', 'confidence',
                           'reward', 'episode', 'step']
    svc_df[out_cols].to_csv(f'xai_obs_{svc}.csv', index=False)
    print(f"  OK xai_obs_{svc}.csv ({len(svc_df)} rows, obs_dim={n_obs})")

print(f"\n{'=' * 65}")
print("Step 1 complete.")
print("=" * 65)
