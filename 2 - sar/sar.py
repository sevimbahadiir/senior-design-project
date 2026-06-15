import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configuration 

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']


SERVICE_WEIGHTS = {
    'cart':      1.5,
    'catalogue': 1.5,
    'shipping':  2.0,
    'ratings':   0.5,
    'user':      1.0,
    'payment':   0.0,
}

POD_COST = 0.05  # reward penalty per pod

#Step 1: Load Clean Data 
print("=" * 65)
print("STEP 1: Loading Clean Data")
print("=" * 65)

df = pd.read_csv('rl_dataset_clean.csv')
print(f"  Loaded : {len(df)} rows | {len(df.columns)} columns")
print(f"  Runs   : {sorted(df['user_load'].unique())}")

state_cols = []
for svc in SERVICES:
    for metric in METRICS:
        col = f'{svc}_{metric}'
        if col in df.columns:
            state_cols.append(col)

print(f"  State  : {len(state_cols)} columns "
      f"({len(SERVICES)} services × {len(METRICS)} metrics)")

#Step 2: Auto-compute p50 Thresholds
print("\n" + "=" * 65)
print("STEP 2: Auto-Computing p50 Thresholds from Dataset")
print("=" * 65)
print("  Thresholds derived from non-zero latency values.")
print("  Saved to reward_config.json for reproducibility.")
print()
print(f"  {'Service':<12} {'p25 (ms)':>10} {'p50 (ms)':>10} "
      f"{'p75 (ms)':>10} {'Weight':>8} {'In Reward':>10}")
print(f"  {'-'*58}")

reward_config = {}
for svc in SERVICES:
    lat_col = f'{svc}_latency'
    valid   = df[df[lat_col] > 0][lat_col]

    p25 = float(valid.quantile(0.25))
    p50 = float(valid.quantile(0.50))
    p75 = float(valid.quantile(0.75))

    reward_config[svc] = {
        'p25':    round(p25, 2),
        'p50':    round(p50, 2),
        'p75':    round(p75, 2),
        'weight': SERVICE_WEIGHTS[svc],
        'in_reward': svc != 'payment',
        'n_valid': int((df[lat_col] > 0).sum()),
    }

    in_reward = "YES" if svc != 'payment' else "NO (Option C)"
    print(f"  {svc:<12} {p25:>10.2f} {p50:>10.2f} "
          f"{p75:>10.2f} {SERVICE_WEIGHTS[svc]:>8.1f} {in_reward:>10}")

# Save reward config for reproducibility
with open('reward_config.json', 'w') as f:
    json.dump(reward_config, f, indent=2)
print(f"\n  ✓ reward_config.json saved")

#Step 3: Compute Actions
print("\n" + "=" * 65)
print("STEP 3: Computing Actions")
print("=" * 65)
print("  Action = sign(pod count diff) within each run.")
print("  groupby('user_load') prevents spurious run-boundary transitions.")

action_cols = []
for svc in SERVICES:
    pod_col    = f'{svc}_num_pods'
    action_col = f'{svc}_action'
    diff = df.groupby('user_load')[pod_col].diff().fillna(0)
    df[action_col] = diff.apply(
        lambda x: -1 if x < 0 else (1 if x > 0 else 0)
    ).astype(int)
    action_cols.append(action_col)

print(f"\n  {'Service':<12} {'Scale Down':>12} {'No Change':>12} {'Scale Up':>10}")
print(f"  {'-'*50}")
for svc, col in zip(SERVICES, action_cols):
    counts = df[col].value_counts()
    total  = len(df)
    down   = counts.get(-1, 0)
    keep   = counts.get(0,  0)
    up     = counts.get(1,  0)
    print(f"  {svc:<12} {down:>8} ({down/total*100:4.1f}%)"
          f" {keep:>8} ({keep/total*100:4.1f}%)"
          f" {up:>6} ({up/total*100:4.1f}%)")

#Step 4: Compute Reward
print("\n" + "=" * 65)
print("STEP 4: Computing Reward (tanh-compressed, auto p50)")
print("=" * 65)
print("  Formula : tanh(1 - latency/p50) * weight")
print("  p50     : auto-computed from dataset (Step 2)")
print("  Excluded: payment (Option C)")
print(f"  Pod cost: -{POD_COST} per pod")

def compute_reward(row: pd.Series) -> float:
    reward = 0.0
    for svc, cfg in reward_config.items():
        if not cfg['in_reward']:
            continue  # payment excluded
        lat = row.get(f'{svc}_latency', 0)
        if lat <= 0:
            continue  # no measurement — no signal
        reward += np.tanh(1.0 - lat / cfg['p50']) * cfg['weight']

    # Pod cost penalty
    total_pods = sum(row.get(f'{svc}_num_pods', 0) for svc in SERVICES)
    reward -= total_pods * POD_COST
    return round(reward, 4)

df['reward'] = df.apply(compute_reward, axis=1)

pos = (df['reward'] > 0).sum()
neg = (df['reward'] < 0).sum()
zer = (df['reward'] == 0).sum()
print(f"\n  Reward statistics:")
print(f"    min      : {df['reward'].min():.4f}")
print(f"    max      : {df['reward'].max():.4f}")
print(f"    mean     : {df['reward'].mean():.4f}")
print(f"    std      : {df['reward'].std():.4f}")
print(f"    positive : {pos:>6} rows ({pos/len(df)*100:.1f}%)")
print(f"    zero     : {zer:>6} rows ({zer/len(df)*100:.1f}%)")
print(f"    negative : {neg:>6} rows ({neg/len(df)*100:.1f}%)")

# Per-service contribution
print(f"\n  Per-service reward contribution (mean, non-zero latency rows):")
print(f"  {'Service':<12} {'Mean':>10} {'Min':>10} {'Max':>10} {'p50 used':>12}")
print(f"  {'-'*58}")
for svc, cfg in reward_config.items():
    if not cfg['in_reward']:
        print(f"  {svc:<12} {'excluded (Option C)':>44}")
        continue
    lat_col = f'{svc}_latency'
    valid   = df[df[lat_col] > 0]
    if len(valid) == 0:
        continue
    contribs = valid[lat_col].apply(
        lambda lat: np.tanh(1.0 - lat / cfg['p50']) * cfg['weight']
    )
    print(f"  {svc:<12} {contribs.mean():>10.4f} "
          f"{contribs.min():>10.4f} {contribs.max():>10.4f} "
          f"{cfg['p50']:>12.2f}")

#Step 5: Build Next State
print("\n" + "=" * 65)
print("STEP 5: Building Next State")
print("=" * 65)
print("  next_state = shift(-1) within each run.")
print("  Last row of each run is dropped (no valid next state).")

next_state_cols = []
for col in state_cols:
    next_col = f'next_{col}'
    df[next_col] = df.groupby('user_load')[col].shift(-1)
    next_state_cols.append(next_col)

before = len(df)
df = df.dropna(subset=next_state_cols).reset_index(drop=True)
print(f"\n  Dropped  : {before - len(df)} rows (last row of each run)")
print(f"  Remaining: {len(df)} rows")

#Step 6: Save Raw SAR
print("\n" + "=" * 65)
print("STEP 6: Saving Raw SAR Dataset")
print("=" * 65)

col_order = (['date', 'user_load'] + state_cols +
             action_cols + ['reward'] + next_state_cols)
df_raw = df[col_order].copy()
df_raw.to_csv('rl_dataset_sar_raw.csv', index=False)

print(f"  ✓ rl_dataset_sar_raw.csv")
print(f"    Rows    : {len(df_raw)}")
print(f"    Columns : {len(df_raw.columns)} "
      f"(2 meta + {len(state_cols)} state + {len(action_cols)} action "
      f"+ 1 reward + {len(next_state_cols)} next_state)")

#Step 7: Normalize
print("\n" + "=" * 65)
print("STEP 7: Normalizing (MinMax, train-only fit, clipped to [0,1])")
print("=" * 65)
print("  Scaler fitted on train split ONLY (80%) — no data leakage.")
print("  Out-of-range values clipped to [0, 1] after transform.")

normalize_cols = state_cols + next_state_cols

# Time-ordered split — no shuffle to preserve temporal structure
train_idx, val_idx = train_test_split(
    df_raw.index, test_size=0.2, shuffle=False
)

scaler = MinMaxScaler()
df_norm = df_raw.copy()

scaler.fit(df_raw.loc[train_idx, normalize_cols])
df_norm[normalize_cols] = scaler.transform(df_raw[normalize_cols])

# Clip out-of-range values
before_clip = ((df_norm[normalize_cols] < 0) |
               (df_norm[normalize_cols] > 1)).any(axis=1).sum()
df_norm[normalize_cols] = df_norm[normalize_cols].clip(0, 1)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

df_norm.to_csv('rl_dataset_sar_normalized.csv', index=False)

print(f"\n  ✓ rl_dataset_sar_normalized.csv")
print(f"  ✓ scaler.pkl")
print(f"    Train rows      : {len(train_idx)}")
print(f"    Val rows        : {len(val_idx)}")
print(f"    Normalized cols : {len(normalize_cols)} (state + next_state)")
print(f"    Clipped rows    : {before_clip} ({before_clip/len(df_norm)*100:.1f}%)")
print(f"    NOT normalized  : action (-1/0/+1), reward (natural scale)")

#Step 8: Verification
print("\n" + "=" * 65)
print("STEP 8: Verification")
print("=" * 65)

min_val = df_norm[normalize_cols].min().min()
max_val = df_norm[normalize_cols].max().max()
print(f"  Normalized range : [{min_val:.6f}, {max_val:.6f}]"
      f"  {'✓' if 0 <= min_val and max_val <= 1.0 else '✗ OUT OF RANGE'}")

action_ok = all(
    set(df_norm[c].unique()).issubset({-1, 0, 1}) for c in action_cols
)
print(f"  Action values    : {{-1, 0, 1}} only  {'✓' if action_ok else '✗'}")

boundary_ok = all(
    all(df_norm[df_norm['user_load'] == load].iloc[0][f'{svc}_action'] == 0
        for svc in SERVICES)
    for load in df_norm['user_load'].unique()
)
print(f"  Run boundaries   : first row actions = 0  "
      f"{'✓' if boundary_ok else '✗'}")

errors = 0
for load in df_norm['user_load'].unique():
    sub = df_raw[df_raw['user_load'] == load].reset_index(drop=True)
    for i in range(len(sub) - 1):
        for col in ['cart_num_pods', 'cart_latency']:
            if not np.isclose(sub.loc[i, f'next_{col}'],
                              sub.loc[i+1, col], atol=1e-6):
                errors += 1
print(f"  State consistency: mismatches = {errors}  "
      f"{'✓' if errors == 0 else '✗'}")

all_ok = (0 <= min_val and max_val <= 1.0 and
          action_ok and boundary_ok and errors == 0)
print(f"\n  Overall: {'ALL CHECKS PASSED ✓' if all_ok else 'SOME CHECKS FAILED ✗'}")

#Summary
print("\n" + "=" * 65)
print("SUMMARY REPORT")
print("=" * 65)
print(f"  Total rows         : {len(df_norm)}")
print(f"  State dimensions   : {len(state_cols)}")
print(f"  Action dimensions  : {len(action_cols)}")
print(f"  Next state dims    : {len(next_state_cols)}")
print(f"  Reward mean        : {df_norm['reward'].mean():.4f}")
print(f"  Reward std         : {df_norm['reward'].std():.4f}")
print(f"  Reward range       : [{df_norm['reward'].min():.4f},"
      f" {df_norm['reward'].max():.4f}]")
print()
print(f"  p50 thresholds     : auto-computed from dataset ✓")
print(f"  reward_config.json : saved for reproducibility ✓")
print()
print(f"  Column structure:")
print(f"    [0]       date")
print(f"    [1]       user_load")
print(f"    [2-31]    state      ({len(state_cols)} cols)")
print(f"    [32-37]   action     ({len(action_cols)} cols)")
print(f"    [38]      reward")
print(f"    [39-68]   next_state ({len(next_state_cols)} cols)")
print()
print(f"  Files saved:")
print(f"    rl_dataset_sar_raw.csv        — unnormalized SAR")
print(f"    rl_dataset_sar_normalized.csv — normalized SAR (DT input)")
print(f"    scaler.pkl                    — fitted MinMaxScaler")
print(f"    reward_config.json            — p50 thresholds (reproducibility)")
print("=" * 65)
print("Done!")