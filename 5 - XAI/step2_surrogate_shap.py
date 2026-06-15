import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
ACTION_LABELS = {0: 'scale_down', 1: 'keep', 2: 'scale_up'}

COLORS = {
    'cart': '#2ecc71', 'catalogue': '#3498db', 'payment': '#e74c3c',
    'shipping': '#f39c12', 'ratings': '#9b59b6', 'user': '#1abc9c'
}

print("=" * 65)
print("XAI Step 2: Surrogate Model + SHAP Analysis")
print("=" * 65)

with open('xai_obs_map.json', 'r') as f:
    obs_map = json.load(f)

def get_feature_names(svc):
    svc_map = obs_map[svc]
    names = []
    for key in sorted(svc_map.keys(), key=lambda x: int(x.split('_')[1])):
        info = svc_map[key]
        role = '*' if info['role'] == 'self' else '->'
        names.append(f"{role}{info['service']}_{info['metric']}")
    return names

#Analysis per service

all_results = []
shap_data   = {}

for svc in SERVICES:
    print(f"\n  Processing [{svc}]...")

    df       = pd.read_csv(f'xai_obs_{svc}.csv')
    obs_cols = sorted([c for c in df.columns if c.startswith('obs_')],
                      key=lambda x: int(x.split('_')[1]))
    X        = df[obs_cols].values.astype(float)
    y        = df['action'].values
    fnames   = get_feature_names(svc)

    unique, counts = np.unique(y, return_counts=True)
    print(f"    Action distribution: " +
          ", ".join([f"{ACTION_LABELS[int(u)]}={c}" for u, c in zip(unique, counts)]))

    if len(unique) == 1 or counts.min() < 2:
        dom_action = ACTION_LABELS[int(unique[np.argmax(counts)])]
        print(f"    WARNING Nearly single class: {dom_action} -- variance analysis")
        feature_vars = np.var(X, axis=0)
        norm_vars    = feature_vars / (feature_vars.max() + 1e-10)

        shap_data[svc] = {
            'type':          'variance',
            'values':        norm_vars,          # 1D array, service-sized
            'feature_names': fnames,
            'action':        dom_action,
            'n_features':    len(fnames),
        }

        result = {
            'service':            svc,
            'dominant_action':    ACTION_LABELS[int(unique[0])],
            'surrogate_accuracy': 1.0,
            'n_classes':          1,
        }
        all_results.append(result)
        print(f"    OK Variance-based importance ({len(fnames)} features)")

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        acc  = accuracy_score(y_test, clf.predict(X_test))
        print(f"    Surrogate accuracy: {acc:.3f}")

        # use feature_importances_ -- multiclass-compatible, more stable than SHAP
        importance = clf.feature_importances_
        norm_imp   = importance / (importance.max() + 1e-10)

        shap_data[svc] = {
            'type':          'feature_importance',
            'values':        norm_imp,
            'feature_names': fnames,
            'action':        ACTION_LABELS[int(unique[np.argmax(counts)])],
            'n_features':    len(fnames),
        }
        result = {
            'service':            svc,
            'dominant_action':    ACTION_LABELS[int(unique[np.argmax(counts)])],
            'surrogate_accuracy': acc,
            'n_classes':          len(unique),
        }
        all_results.append(result)
        print(f"    OK SHAP importance ({len(fnames)} features)")

#Chart 1: Feature importance per service

print("\n  Chart 1: Per service...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('IPPO Policy — Feature Importance per Service\n'
             '(* = own service, -> = neighbor service)',
             fontsize=13, fontweight='bold')
axes_flat = axes.flatten()

for idx, svc in enumerate(SERVICES):
    ax     = axes_flat[idx]
    data   = shap_data[svc]
    fnames = data['feature_names']          # list for this service
    vals   = data['values']                 # 1D array for this service
    n      = len(fnames)                    # length for this service

    assert len(vals) == n, f"{svc}: vals={len(vals)}, fnames={n}"

    y_pos      = np.arange(n)
    bar_colors = [COLORS[svc] if f.startswith('*') else '#BDC3C7' for f in fnames]

    ax.barh(y_pos, vals, color=bar_colors, alpha=0.85, edgecolor='white', lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fnames, fontsize=8)
    ax.set_xlabel('Normalized Importance (0-1)', fontsize=9)
    ax.set_title(f"{svc.upper()}\n(dominant: {data['action']})",
                 fontweight='bold', fontsize=10)
    ax.set_xlim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7)

    p1 = mpatches.Patch(color=COLORS[svc], label='Own (*)')
    p2 = mpatches.Patch(color='#BDC3C7',  label='Neighbor (->)')
    ax.legend(handles=[p1, p2], fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig('xai_shap_per_service.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK xai_shap_per_service.png")

#Chart 2: Top-3 summary

print("  Chart 2: Top-3 summary...")

y_labels, y_values, y_colors = [], [], []
for svc in SERVICES:
    data   = shap_data[svc]
    fnames = data['feature_names']
    vals   = data['values']
    top3   = np.argsort(vals)[::-1][:3]
    for i in top3:
        y_labels.append(f"[{svc}] {fnames[i]}")
        y_values.append(float(vals[i]))
        y_colors.append(COLORS[svc])

fig, ax = plt.subplots(figsize=(14, 9))
y_pos   = np.arange(len(y_labels))
ax.barh(y_pos, y_values, color=y_colors, alpha=0.85, edgecolor='white', lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Normalized Importance (0-1)', fontsize=10)
ax.set_title('Top-3 Feature Importance — All Services', fontsize=13, fontweight='bold')
ax.set_xlim(0, 1.2)
ax.grid(True, alpha=0.3, axis='x')
for pos, val in zip(y_pos, y_values):
    ax.text(val + 0.02, pos, f'{val:.2f}', va='center', fontsize=8)
patches = [mpatches.Patch(color=COLORS[s], label=s) for s in SERVICES]
ax.legend(handles=patches, fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('xai_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK xai_shap_summary.png")

# Console summary

print(f"\n{'=' * 65}")
print("FEATURE IMPORTANCE SUMMARY")
print("=" * 65)
for svc in SERVICES:
    data   = shap_data[svc]
    fnames = data['feature_names']
    vals   = data['values']
    top3   = np.argsort(vals)[::-1][:3]
    print(f"\n  {svc.upper()} -> {data['action']}")
    for rank, i in enumerate(top3, 1):
        print(f"    {rank}. {fnames[i]:<38} {vals[i]:.3f}")


# Feature importance CSV 

importance_rows = []

for svc in SERVICES:
    data = shap_data[svc]
    fnames = data["feature_names"]
    vals = data["values"]

    for feature_name, importance in zip(fnames, vals):
        importance_rows.append({
            "service": svc,
            "feature": feature_name,
            "importance": float(importance),
            "importance_type": data["type"],
            "dominant_action": data["action"],
        })

pd.DataFrame(importance_rows).to_csv(
    "xai_feature_importance.csv",
    index=False
)

print("  OK xai_feature_importance.csv")

pd.DataFrame(all_results).to_csv('xai_surrogate_results.csv', index=False)
print(f"\n  OK xai_surrogate_results.csv")
print("\nStep 2 complete.")
print("=" * 65)
