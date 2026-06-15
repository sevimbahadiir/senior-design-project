import pickle
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#Configuration

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

HIDDEN_DIMS  = [256, 256, 128]
DROPOUT      = 0.2
BATCH_SIZE   = 64
EPOCHS       = 300
LR           = 1e-3
LR_PATIENCE  = 25  
LR_FACTOR    = 0.5
ES_PATIENCE  = 60   
HUBER_DELTA  = 1.0

#Step 1: Load Data
print("\n" + "=" * 65)
print("STEP 1: Loading Data")
print("=" * 65)

df = pd.read_csv('rl_dataset_sar_normalized.csv')
print(f"  Loaded: {len(df)} rows | {len(df.columns)} columns")
print(f"  (Old dataset: 14,390 rows -> New: {len(df)} rows, +{(len(df)-14390)/14390*100:.0f}%)")

state_cols      = [f'{svc}_{m}' for svc in SERVICES for m in METRICS
                   if f'{svc}_{m}' in df.columns]
action_cols     = [f'{svc}_action' for svc in SERVICES]
next_state_cols = [f'next_{c}' for c in state_cols]

input_dim  = len(state_cols) + len(action_cols)
output_dim = len(next_state_cols)

print(f"  State dim  : {len(state_cols)}")
print(f"  Action dim : {len(action_cols)}")
print(f"  Input dim  : {input_dim}")
print(f"  Output dim : {output_dim}")

#Step 2: Prepare Tensors
print("\n" + "=" * 65)
print("STEP 2: Preparing Train / Validation Split")
print("=" * 65)

X = df[state_cols + action_cols].values.astype(np.float32)
y = df[next_state_cols].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"  Train : {len(X_train)} rows")
print(f"  Val   : {len(X_val)} rows")

assert X_train[:, len(state_cols):].min() >= -1
assert X_train[:, len(state_cols):].max() <=  1
print(f"  Action range check: [-1, 1] OK")

#Dataset / DataLoader 
class TransitionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    TransitionDataset(X_train, y_train),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TransitionDataset(X_val, y_val),
    batch_size=BATCH_SIZE, shuffle=False
)

#  Model Architecture 
print("\n" + "=" * 65)
print("STEP 3: Model Architecture")
print("=" * 65)

class DigitalTwin(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model       = DigitalTwin(input_dim, output_dim, HIDDEN_DIMS, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Architecture : {input_dim} -> {' -> '.join(map(str, HIDDEN_DIMS))} -> {output_dim}")
print(f"  Total params : {total_params:,}")
print(f"  Trainable    : {trainable:,}")
print(f"  Dropout      : {DROPOUT}")

# Training
print("\n" + "=" * 65)
print("STEP 4: Training")
print("=" * 65)
print(f"  Loss      : Huber (delta={HUBER_DELTA})")
print(f"  Optimizer : Adam (lr={LR})")
print(f"  LR decay  : ReduceLROnPlateau (patience={LR_PATIENCE}, factor={LR_FACTOR})")
print(f"  Early stop: patience={ES_PATIENCE} epochs  (v1: 30)")
print(f"  Epochs    : up to {EPOCHS}")
print()

criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR
)

train_losses = []
val_losses   = []
lr_history   = []
best_val_loss = float('inf')
best_epoch    = 0
es_counter    = 0
train_start   = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    epoch_train_loss /= len(train_loader)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            epoch_val_loss += loss.item()
    epoch_val_loss /= len(val_loader)

    scheduler.step(epoch_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    lr_history.append(current_lr)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch    = epoch
        es_counter    = 0
        torch.save(model.state_dict(), 'digital_twin_best.pth')
    else:
        es_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - train_start
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {epoch_train_loss:.6f} | "
              f"Val: {epoch_val_loss:.6f} | "
              f"Best: {best_val_loss:.6f} (ep {best_epoch}) | "
              f"LR: {current_lr:.2e} | "
              f"ES: {es_counter}/{ES_PATIENCE} | "
              f"{elapsed:.0f}s")

    if es_counter >= ES_PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}.")
        break

torch.save(model.state_dict(), 'digital_twin_final.pth')
print(f"\n  Training complete.")
print(f"  Best val loss : {best_val_loss:.6f} at epoch {best_epoch}")
print(f"  Total time    : {time.time() - train_start:.1f}s")

# Evaluation 
print("\n" + "=" * 65)
print("STEP 5: Evaluation")
print("=" * 65)

model.load_state_dict(torch.load('digital_twin_best.pth', map_location=device))
model.eval()

with torch.no_grad():
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_pred  = model(X_val_t).cpu().numpy()

mae  = np.mean(np.abs(y_pred - y_val))
rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
r2   = 1 - np.sum((y_pred - y_val) ** 2) / np.sum((y_val - y_val.mean()) ** 2)

print(f"  MAE  : {mae:.6f}  (previous: 0.041306)")
print(f"  RMSE : {rmse:.6f}  (previous: 0.066100)")
print(f"  R^2  : {r2:.6f}  (previous: 0.906600)")
print()
print(f"  Result: {'IMPROVED' if mae < 0.041 else 'SAME LEVEL' if mae < 0.045 else 'NEEDS REVIEW'}")

print(f"\n  Per-service next_state MAE:")
print(f"  {'Service':<12} {'num_pods':>10} {'cpu':>10} {'mem':>10} {'rps':>10} {'latency':>10}")
print(f"  {'-'*62}")
for i, svc in enumerate(SERVICES):
    base   = i * len(METRICS)
    errors = [np.mean(np.abs(y_pred[:, base+j] - y_val[:, base+j]))
              for j in range(len(METRICS))]
    print(f"  {svc:<12} " + " ".join(f"{e:>10.6f}" for e in errors))

print(f"\n  Per-service latency R^2:")
print(f"  {'Service':<12} {'R^2':>10} {'MAE':>10}  Result")
print(f"  {'-'*50}")
old_r2 = {'cart':0.863,'catalogue':0.877,'payment':0.437,'shipping':0.834,'ratings':0.816,'user':0.786}
for i, svc in enumerate(SERVICES):
    lat_idx = i * len(METRICS) + 4
    pred_l, true_l = y_pred[:, lat_idx], y_val[:, lat_idx]
    ss_res  = np.sum((pred_l - true_l)**2)
    ss_tot  = np.sum((true_l - true_l.mean())**2)
    r2_svc  = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    mae_l   = np.mean(np.abs(pred_l - true_l))
    delta   = r2_svc - old_r2[svc]
    arrow   = 'UP' if delta > 0.01 else ('DOWN' if delta < -0.01 else 'SAME')
    print(f"  {svc:<12} {r2_svc:>10.4f} {mae_l:>10.5f}  {arrow} ({delta:+.3f} vs previous)")

# Compounding Error Analysis (5-step rollout) 
print(f"\n" + "=" * 65)
print("STEP 6: Compounding Error Analysis (5-step autoregressive rollout)")
print("=" * 65)
print("  Advancing 5 steps from 100 random starting points.")
print("  Previous DT: step1=0.040, step5=0.091 (2.3x increase)")

np.random.seed(42)
starts = np.random.choice(len(X_val) - 6, 100, replace=False)
rollout_maes = {k: [] for k in range(1, 6)}

for start in starts:
    state = X_val[start].copy()
    for step in range(1, 6):
        inp = torch.tensor(state[None], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(inp).cpu().numpy()[0]
        pred = np.clip(pred, 0, 1)
        true_ns = y_val[start + step - 1]
        rollout_maes[step].append(np.mean(np.abs(pred - true_ns)))
        state[:30] = pred

print(f"\n  {'Step':>6} {'MAE':>10} {'Previous MAE':>12} {'Change':>10}")
print(f"  {'-'*42}")
old_rollout = {1:0.040, 2:0.056, 3:0.068, 4:0.081, 5:0.091}
rollout_records = []
for step in range(1, 6):
    m    = np.mean(rollout_maes[step])
    old  = old_rollout[step]
    diff = m - old
    arrow = 'WORSE' if diff > 0.005 else ('BETTER' if diff < -0.005 else 'SAME')
    print(f"  {step:>6} {m:>10.5f} {old:>12.5f} {diff:>+9.5f}  {arrow}")
    rollout_records.append({'step': step, 'mae': m, 'old_mae': old, 'delta': diff})

pd.DataFrame(rollout_records).to_csv('dt_rollout_error.csv', index=False)
print(f"\n  Saved: dt_rollout_error.csv")

#Training Log 
pd.DataFrame({
    'epoch':      range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss':   val_losses,
    'lr':         lr_history,
}).to_csv('dt_training_log.csv', index=False)

#  
#Plot 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curve
axes[0].plot(train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
axes[0].plot(val_losses,   label='Val Loss',   color='darkorange', linewidth=1.5)
axes[0].axvline(best_epoch - 1, color='green', linestyle='--',
                alpha=0.7, label=f'Best (ep {best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Huber Loss')
axes[0].set_title('Digital Twin — Training Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Latency scatter
lat_indices = [i * len(METRICS) + 4 for i in range(len(SERVICES))]
y_pred_lat  = y_pred[:, lat_indices].flatten()
y_true_lat  = y_val[:,  lat_indices].flatten()
sample_idx  = np.random.choice(len(y_pred_lat), min(2000, len(y_pred_lat)), replace=False)
axes[1].scatter(y_true_lat[sample_idx], y_pred_lat[sample_idx],
                alpha=0.3, s=5, color='steelblue')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect prediction')
axes[1].set_xlabel('Actual next_latency (normalized)')
axes[1].set_ylabel('Predicted next_latency (normalized)')
axes[1].set_title(f'Latency Prediction (R^2={r2:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Compounding error
steps     = list(rollout_maes.keys())
maes_new  = [np.mean(rollout_maes[s]) for s in steps]
maes_old  = [old_rollout[s] for s in steps]
axes[2].plot(steps, maes_old, 'o--', color='darkorange', linewidth=2, label='Old DT')
axes[2].plot(steps, maes_new, 's-',  color='steelblue',  linewidth=2, label='New DT')
axes[2].set_xlabel('Rollout Step')
axes[2].set_ylabel('MAE (normalized)')
axes[2].set_title('Compounding Error Comparison')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(steps)

plt.tight_layout()
plt.savefig('dt_training_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: dt_training_plot.png")

# ─── Scaler ───────────────────────────────────────────────────────────────────
shutil.copy('scaler.pkl', 'dt_scaler.pkl')
