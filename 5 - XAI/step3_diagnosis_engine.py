import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

MARL_IMPROVEMENT = {
    'cart':      {'rb_lat': 187.0,   'ippo_lat': 110.0,  'improved': True},
    'catalogue': {'rb_lat': 119.1,   'ippo_lat':  55.0,  'improved': True},
    'payment':   {'rb_lat': 696.2,   'ippo_lat': 714.2,  'improved': False},
    'shipping':  {'rb_lat': 17492.9, 'ippo_lat': 19795.8,'improved': False},
    'ratings':   {'rb_lat': 1112.2,  'ippo_lat':  903.0, 'improved': True},
    'user':      {'rb_lat':  47.3,   'ippo_lat':   32.7, 'improved': True},
}

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]

def denorm(svc, metric, val):
    idx  = state_cols.index(f'{svc}_{metric}')
    dmin = scaler.data_min_[idx]
    dmax = scaler.data_max_[idx]
    return float(val * (dmax - dmin) + dmin)

df = pd.read_csv('xai_behavior_dataset.csv')
with open('xai_obs_map.json') as f:
    obs_map = json.load(f)
with open('reward_config.json') as f:
    reward_cfg = json.load(f)

print("=" * 65)
print("XAI Step 3: Rule-based Diagnosis Engine (v3)")
print("=" * 65)
print("  v3 change: Shipping bottleneck detection is action-independent")

def get_service_metrics(svc):
    svc_df = df[df['service'] == svc]
    self_metrics = {
        info['metric']: f"obs_{key.split('_')[1]}"
        for key, info in obs_map[svc].items()
        if info['role'] == 'self'
    }
    result = {}
    for metric, col in self_metrics.items():
        if col in svc_df.columns:
            norm_val = svc_df[col].mean()
            real_val = denorm(svc, metric, norm_val)
            result[metric] = {'norm': norm_val, 'real': real_val}
    result['action']     = svc_df['action_label'].mode()[0]
    result['confidence'] = svc_df['confidence'].mean()
    return result

def diagnose(svc, metrics, action):
    evidence = []
    details  = {}

    lat_norm = metrics.get('latency',      {}).get('norm', 0)
    cpu_norm = metrics.get('cpu_usage',    {}).get('norm', 0)
    pod_norm = metrics.get('num_pods',     {}).get('norm', 0)
    rps_norm = metrics.get('request_rate', {}).get('norm', 0)
    lat_real = metrics.get('latency',      {}).get('real', 0)
    pod_real = metrics.get('num_pods',     {}).get('real', 0)

    p50_real  = reward_cfg.get(svc, {}).get('p50', 1000)
    in_reward = reward_cfg.get(svc, {}).get('in_reward', True)
    improved  = MARL_IMPROVEMENT[svc]['improved']
    ippo_lat  = MARL_IMPROVEMENT[svc]['ippo_lat']
    rb_lat    = MARL_IMPROVEMENT[svc]['rb_lat']

    details['latency_ms']     = round(lat_real, 1)
    details['p50_ms']         = p50_real
    details['latency_ratio']  = round(lat_real / p50_real, 2) if p50_real > 0 else 0
    details['pod_count']      = round(pod_real, 1)
    details['cpu_norm']       = round(cpu_norm, 3)
    details['action']         = action
    details['in_reward']      = in_reward
    details['ippo_lat_ms']    = ippo_lat
    details['rb_lat_ms']      = rb_lat
    details['pct_improvement']= round((rb_lat - ippo_lat) / rb_lat * 100, 1) if rb_lat > 0 else 0
    details['confidence']     = round(metrics.get('confidence', 0), 3)

    pct = details['pct_improvement']

    #  Rule 0: Excluded from reward
    if not in_reward:
        evidence.append("Service excluded from reward (Option C)")
        evidence.append(f"Latency: {lat_real:.0f}ms -- monitored but not optimized")
        return 'reward_excluded', 'info', evidence, details


    if improved and pct > 10.0:
        evidence.append(f"MARL latency improvement: {rb_lat:.0f}ms -> {ippo_lat:.0f}ms ({pct:+.1f}%)")
        evidence.append(f"Current policy: {action} (confidence: {details.get('confidence', 0):.2f})")
        return 'marl_optimized', 'ok', evidence, details

    if svc == 'shipping' and not improved and lat_real > p50_real:
        evidence.append(f"Latency {lat_real:.0f}ms > p50 ({p50_real:.0f}ms)")
        evidence.append(f"Action '{action}' was selected but latency did not improve")
        evidence.append(f"MARL vs Rule-based: {ippo_lat:.0f}ms vs {rb_lat:.0f}ms "
                        f"({ippo_lat-rb_lat:+.0f}ms, {abs(pct):.1f}% worsening)")
        evidence.append("Pod change (increase or decrease) has no effect -- structural MySQL/IO bottleneck")
        evidence.append(f"CPU normalized: {cpu_norm:.2f} -- I/O-bound, adding pods is not the solution")
        return 'pod_insensitive_bottleneck', 'critical', evidence, details

    if lat_real > 1.5 * p50_real and action == 'scale_up' and cpu_norm < 0.4:
        evidence.append(f"Latency {lat_real:.0f}ms > 1.5x p50 ({1.5*p50_real:.0f}ms)")
        evidence.append(f"CPU low ({cpu_norm:.2f}) -> not CPU-bound")
        evidence.append("Scale-up was selected but there is no CPU bottleneck")
        return 'pod_insensitive_bottleneck', 'warning', evidence, details

    if lat_real > 1.2 * p50_real and action in ('keep', 'scale_down'):
        evidence.append(f"Latency {lat_real:.0f}ms > 1.2x p50 ({1.2*p50_real:.0f}ms)")
        evidence.append(f"Agent chooses {action} -- non-reactive policy")
        return 'under_provisioned', 'warning', evidence, details

    if lat_real < 0.5 * p50_real and action == 'scale_down':
        evidence.append(f"Latency {lat_real:.0f}ms < 0.5x p50 ({0.5*p50_real:.0f}ms)")
        evidence.append("Efficient resource usage -- scale_down is appropriate")
        return 'over_provisioned', 'ok', evidence, details

    if lat_real <= p50_real:
        evidence.append(f"Latency {lat_real:.0f}ms <= p50 ({p50_real:.0f}ms)")
        evidence.append(f"Agent {action} -- normal operation")
        return 'healthy', 'ok', evidence, details

    evidence.append(f"Latency {lat_real:.0f}ms, p50 {p50_real:.0f}ms, action {action}")
    return 'monitoring_required', 'info', evidence, details



SEV_ICON = {'critical': '[CRIT]', 'warning': '[WARN]', 'ok': '[OK]', 'info': '[INFO]'}
SEV_COLORS = {
    'critical': '#e74c3c', 'warning': '#f39c12',
    'ok': '#2ecc71',       'info':    '#3498db',
}
DIAG_TR = {
    'pod_insensitive_bottleneck': 'Pod-Insensitive Bottleneck',
    'under_provisioned':          'Under-Provisioned',
    'over_provisioned':           'Over-Provisioned (Efficient)',
    'marl_optimized':             'MARL Optimized OK',
    'healthy':                    'Healthy',
    'reward_excluded':            'Excluded from Reward (Option C)',
    'monitoring_required':        'Monitoring Required',
}

print(f"\n  {'Service':<12} {'Diagnosis':<32} {'Sev':<5} {'Latency':>10} {'p50':>8} {'Improvement':>12}")
print(f"  {'-'*82}")

diagnosis_report = {}
rows = []

for svc in SERVICES:
    metrics = get_service_metrics(svc)
    action  = metrics['action']
    diagnosis, severity, evidence, details = diagnose(svc, metrics, action)
    diagnosis_report[svc] = {
        'diagnosis':  diagnosis,
        'severity':   severity,
        'action':     action,
        'confidence': round(metrics['confidence'], 3),
        'evidence':   evidence,
        'details':    details,
    }
    icon    = SEV_ICON.get(severity, '[?]')
    pct     = details['pct_improvement']
    pct_str = f"{pct:+.1f}%" if pct != 0 else "--"
    print(f"  {svc:<12} {DIAG_TR.get(diagnosis, diagnosis):<32} "
          f"{icon} {severity:<3} {details['latency_ms']:>10.1f} "
          f"{details['p50_ms']:>8.1f} {pct_str:>12}")
    rows.append({
        'service':        svc,
        'diagnosis':      diagnosis,
        'severity':       severity,
        'action':         action,
        'latency_ms':     details['latency_ms'],
        'p50_ms':         details['p50_ms'],
        'latency_ratio':  details['latency_ratio'],
        'pod_count':      details['pod_count'],
        'cpu_norm':       details['cpu_norm'],
        'confidence':     round(metrics['confidence'], 3),
        'pct_improvement': details['pct_improvement'],
    })

# ─── Detailed Output ───────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print("DETAILED DIAGNOSIS")
print("=" * 65)

for svc, report in diagnosis_report.items():
    d    = report['details']
    icon = SEV_ICON.get(report['severity'], '[?]')
    print(f"\n  -- {svc.upper()} {icon} --")
    print(f"  Diagnosis  : {DIAG_TR.get(report['diagnosis'], report['diagnosis'])}")
    print(f"  Severity   : {report['severity']}")
    print(f"  Action     : {report['action']} (confidence: {report['confidence']:.3f})")
    print(f"  Latency    : {d['latency_ms']:.1f}ms (p50: {d['p50_ms']}ms, {d['latency_ratio']:.2f}x)")
    if d['pct_improvement'] != 0:
        print(f"  MARL Impr. : {d['rb_lat_ms']:.0f}ms -> {d['ippo_lat_ms']:.0f}ms "
              f"({d['pct_improvement']:+.1f}%)")
    for ev in report['evidence']:
        print(f"    - {ev}")


df_rows = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Rule-based Diagnosis Engine (v3) — Per-Service Analysis\n'
             'IPPO v4 + New DT', fontsize=13, fontweight='bold')

ax = axes[0]
x    = np.arange(len(SERVICES))
w    = 0.35
clrs = [SEV_COLORS[diagnosis_report[s]['severity']] for s in SERVICES]
ax.bar(x - w/2, df_rows['latency_ms'], w, label='IPPO v4 Latency (ms)',
       color=clrs, alpha=0.85, edgecolor='black', lw=0.8)
ax.bar(x + w/2, df_rows['p50_ms'],     w, label='p50 threshold (ms)',
       color='lightgray', alpha=0.85, edgecolor='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(SERVICES, rotation=15)
ax.set_yscale('log')
ax.set_ylabel('Latency (ms, log scale)')
ax.set_title('IPPO v4 Latency vs p50 Threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
diag_labels = [DIAG_TR.get(diagnosis_report[s]['diagnosis'], '?') for s in SERVICES]
conf_vals   = [diagnosis_report[s]['confidence'] for s in SERVICES]
clrs2       = [SEV_COLORS[diagnosis_report[s]['severity']] for s in SERVICES]
bars = ax.barh(SERVICES, conf_vals, color=clrs2, alpha=0.85, edgecolor='black', lw=0.8)
ax.set_xlabel('Decision Confidence')
ax.set_title('Diagnosis & Decision Confidence')
ax.set_xlim(0, 1.35)
ax.grid(True, alpha=0.3, axis='x')
for bar, label, val in zip(bars, diag_labels, conf_vals):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{label}\n({val:.2f})', va='center', fontsize=8)
patches = [mpatches.Patch(color=c, label=l)
           for c, l in [('#e74c3c', 'Critical'), ('#f39c12', 'Warning'),
                        ('#2ecc71', 'OK'),        ('#3498db', 'Info')]]
ax.legend(handles=patches, fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('xai_diagnosis_plot.png', dpi=150, bbox_inches='tight')
plt.close()

with open('xai_diagnosis_report.json', 'w', encoding='utf-8') as f:
    json.dump(diagnosis_report, f, indent=2, ensure_ascii=False)
df_rows.to_csv('xai_diagnosis_summary.csv', index=False)

print(f"\n  OK xai_diagnosis_report.json")
print(f"  OK xai_diagnosis_summary.csv")
print(f"  OK xai_diagnosis_plot.png")
print(f"\n{'=' * 65}")
print("Step 3 complete.")
print("=" * 65)
