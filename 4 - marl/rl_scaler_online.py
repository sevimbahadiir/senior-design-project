"""
rl_scaler_online.py
-------------------
IPPO v5 — Online Fine-Tuning ile Gerçek Kubernetes Cluster'da Pod Scaling

Offline eğitimden farkı:
  - Gerçek Prometheus metriklerinden reward hesaplanır
  - Her UPDATE_INTERVAL adımda PPO update yapılır
  - Policy gerçek cluster davranışına adapte olur
  - Collapsed policy kırılır: gerçek state çeşitliliği keşfi zorlar

Akış:
  1. Prometheus'tan state oku
  2. Aksiyon seç (temperature sampling + pod maskeleme)
  3. kubectl scale et
  4. STEP_INTERVAL saniye bekle
  5. Yeni state oku → gerçek reward hesapla
  6. Buffer'a ekle → her UPDATE_INTERVAL adımda PPO update
  7. Checkpoint kaydet → başa dön

Kullanım:
    python rl_scaler_online.py

Gerekli dosyalar (aynı klasörde):
    checkpoints_v5/ippo_v5_final.pth
    ippo_agent.py
    scaler.pkl
    reward_config.json

Ayarlar:
    DRY_RUN=True ile başla — kubectl çalışmaz, sadece ne yapacağını gösterir
    Her şey normal görününce DRY_RUN=False yap
"""

import os
import sys
import time
import json
import pickle
import logging
import subprocess
import numpy as np
import requests
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CONFIG = {
    'PROMETHEUS_URL'   : 'http://localhost:9091',
    'NAMESPACE'        : 'robot-shop',
    'CHECKPOINT'       : 'checkpoints_v5/ippo_v5_final.pth',
    'ONLINE_CKPT_DIR'  : 'checkpoints_online',   # online update'ler buraya kaydedilir
    'SCALER_PATH'      : 'scaler.pkl',
    'REWARD_CFG'       : 'reward_config.json',
    'STEP_INTERVAL'    : 90,       # saniye — scaling sonrası bekleme
    'TEMPERATURE'      : 1.2,      # offline'dan biraz yüksek — daha fazla keşif
    'METRIC_WINDOW'    : '2m',
    'UPDATE_INTERVAL'  : 5,        # kaç adımda bir PPO update
    'SAVE_INTERVAL'    : 20,       # kaç adımda bir checkpoint kaydet
    'ENTROPY_COEF'     : 0.05,     # online fine-tuning için yüksek entropy — keşif
    'N_STEPS'          : 100,     # None = sonsuz, int = kaç adım
    'DRY_RUN'          : False,     # İLK ÇALIŞTIRMADA True BIRAK
    'LOG_FILE'         : 'rl_scaler_online.log',

    # ── XAI Hook ─────────────────────────────────────────────────────────────
    # Her SAVE_INTERVAL adımda XAI pipeline arka planda çalışır.
    # XAI_SCRIPT_DIR : xai_runner.py'nin bulunduğu XAIguncel klasörü
    # XAI_OUTPUT_DIR : xai_results/ çıktı klasörü (step20, step40, ... alt klasörler)
    # XAI_ENABLED    : False yaparak devre dışı bırakabilirsin
    # Not: Klasör yapısı → Marl 5/ ve XAIguncel/ aynı dizinde olmalı
    'XAI_ENABLED'      : True,
    'XAI_SCRIPT_DIR'   : '../XAIguncel',
    'XAI_OUTPUT_DIR'   : '../XAIguncel/xai_results',
}

# ─── Sabitler ────────────────────────────────────────────────────────────────

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

DEPLOYMENTS = {
    'cart': 'cart', 'catalogue': 'catalogue', 'payment': 'payment',
    'shipping': 'shipping', 'ratings': 'ratings', 'user': 'user',
}

POD_BOUNDS = {
    'cart':      {'min': 1, 'max': 8},
    'catalogue': {'min': 1, 'max': 8},
    'payment':   {'min': 2, 'max': 8},
    'shipping':  {'min': 2, 'max': 8},
    'ratings':   {'min': 1, 'max': 8},
    'user':      {'min': 1, 'max': 8},
}

ACTION_MAP    = {0: -1, 1: 0, 2: 1}
ACTION_LABELS = {0: 'scale_down', 1: 'keep', 2: 'scale_up'}

AGENT_NEIGHBORS = {
    'cart':      ['catalogue', 'shipping'],
    'catalogue': ['ratings', 'cart'],
    'payment':   ['shipping'],
    'shipping':  ['cart', 'payment'],
    'ratings':   ['catalogue'],
    'user':      ['cart'],
}

# Reward ağırlıkları — sar2.py ile aynı
SERVICE_WEIGHTS = {
    'cart': 1.5, 'catalogue': 1.5, 'payment': 0.0,
    'shipping': 2.0, 'ratings': 0.5, 'user': 1.0,
}

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w',
                                   encoding='utf-8', closefd=False)),
        logging.FileHandler(CONFIG['LOG_FILE'], encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)

# ─── Model Yükleme ───────────────────────────────────────────────────────────

def load_agents(checkpoint_path: str):
    """IPPO v5 ajanlarını yükle — optimizer state'leri de yükle (fine-tuning için)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ippo_agent import IPPOAgent

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    obs_dims = {
        'cart': 15, 'catalogue': 15, 'payment': 10,
        'shipping': 15, 'ratings': 10, 'user': 10,
    }

    agents = {}
    for svc in SERVICES:
        agent = IPPOAgent(obs_dim=obs_dims[svc], n_actions=3, service=svc)
        agent.actor.load_state_dict(ckpt['agents'][svc]['actor'])
        agent.critic.load_state_dict(ckpt['agents'][svc]['critic'])
        # Optimizer state'leri de yükle — fine-tuning momentum korunur
        try:
            agent.actor_opt.load_state_dict(ckpt['agents'][svc]['actor_opt'])
            agent.critic_opt.load_state_dict(ckpt['agents'][svc]['critic_opt'])
        except Exception:
            log.warning(f"  {svc}: optimizer state yüklenemedi, sıfırdan başlıyor")
        agent.actor.train()   # fine-tuning için train mode
        agent.critic.train()
        agents[svc] = agent

    log.info(f"Model yüklendi (fine-tuning modu): {checkpoint_path}")
    return agents


def save_checkpoint(agents: dict, step: int, total_reward: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'online_step{step}.pth')
    torch.save({
        'step':         step,
        'total_reward': total_reward,
        'agents': {svc: {
            'actor':      agents[svc].actor.state_dict(),
            'critic':     agents[svc].critic.state_dict(),
            'actor_opt':  agents[svc].actor_opt.state_dict(),
            'critic_opt': agents[svc].critic_opt.state_dict(),
        } for svc in SERVICES},
    }, path)
    log.info(f"  ✓ Checkpoint kaydedildi: {path}")


def load_scaler(scaler_path: str):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


def load_reward_cfg(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── Prometheus ──────────────────────────────────────────────────────────────

def query_prometheus(url: str, query: str):
    try:
        resp = requests.get(f'{url}/api/v1/query',
                            params={'query': query}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get('data', {}).get('result', [])
        if results:
            return float(results[0]['value'][1])
        return None
    except Exception as e:
        log.warning(f"Prometheus hatası: {query[:50]}... → {e}")
        return None


def get_current_pods(namespace: str, deployment: str):
    try:
        r = subprocess.run(
            ['kubectl', 'get', 'deployment', deployment,
             '-n', namespace, '-o', 'jsonpath={.spec.replicas}'],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip())
    except Exception as e:
        log.warning(f"kubectl hatası ({deployment}): {e}")
    return None


def collect_metrics(prometheus_url: str, namespace: str, window: str) -> dict:
    metrics = {}
    for svc in SERVICES:
        m = {}
        pods = get_current_pods(namespace, DEPLOYMENTS[svc])
        m['num_pods'] = float(pods) if pods else 2.0

        cpu = query_prometheus(prometheus_url,
            f'sum(rate(container_cpu_usage_seconds_total{{'
            f'namespace="{namespace}",container="{svc}"}}[{window}]))')
        m['cpu_usage'] = cpu or 0.0

        mem = query_prometheus(prometheus_url,
            f'sum(container_memory_working_set_bytes{{'
            f'namespace="{namespace}",container="{svc}"}}) / 1048576')
        m['mem_usage'] = mem or 0.0

        rps = query_prometheus(prometheus_url,
            f'sum(rate(istio_requests_total{{'
            f'destination_workload_namespace="{namespace}",'
            f'destination_workload=~"{svc}.*",reporter="destination"}}[{window}]))')
        m['request_rate'] = rps or 0.0

        lat = query_prometheus(prometheus_url,
            f'histogram_quantile(0.50,sum(rate('
            f'istio_request_duration_milliseconds_bucket{{'
            f'destination_workload_namespace="{namespace}",'
            f'destination_workload=~"{svc}.*",reporter="destination"}}[{window}]))'
            f'by(le))')
        m['latency'] = lat or 0.0

        metrics[svc] = m
    return metrics


# ─── State / Obs ─────────────────────────────────────────────────────────────

# Manuel normalizasyon sinirlari — SAR dataseti istatistiklerinden
# (SAR scaler'i 60 feature uzerine fit edilmis, live 30-feature state uyumsuz)
METRIC_BOUNDS = {
    'num_pods':     {'min': 1.0,   'max': 10.0},
    'cpu_usage':    {'min': 0.0,   'max': 2.0},
    'mem_usage':    {'min': 0.0,   'max': 512.0},
    'request_rate': {'min': 0.0,   'max': 50.0},
    'latency':      {'min': 0.0,   'max': 30000.0},
}

def build_global_state(raw_metrics: dict, scaler=None) -> np.ndarray:
    """
    Ham metriklerden global state vektoru olustur — 30 boyut.
    Manuel min-max normalizasyon kullanilir (SAR scaler bypass).
    """
    row = []
    for svc in SERVICES:
        for metric in METRICS:
            val = raw_metrics[svc][metric]
            lo  = METRIC_BOUNDS[metric]['min']
            hi  = METRIC_BOUNDS[metric]['max']
            row.append(float(np.clip((val - lo) / (hi - lo + 1e-8), 0.0, 1.0)))
    arr = np.array(row, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def build_obs(global_state: np.ndarray) -> dict:
    service_idx = {svc: i * len(METRICS) for i, svc in enumerate(SERVICES)}
    obs = {}
    for svc in SERVICES:
        svc_obs = list(global_state[service_idx[svc]: service_idx[svc] + len(METRICS)])
        for neighbor in AGENT_NEIGHBORS[svc]:
            ni = service_idx[neighbor]
            svc_obs.extend(global_state[ni: ni + len(METRICS)])
        obs[svc] = np.array(svc_obs, dtype=np.float32)
    return obs


# ─── Reward (Gerçek Cluster) ─────────────────────────────────────────────────

def compute_reward(raw_metrics: dict, reward_cfg: dict, scaler,
                   prev_pods: dict, new_pods: dict) -> float:
    """
    Gerçek Prometheus metriklerinden tanh reward hesapla.
    SAR aşamasındaki formülle aynı:
        r = sum[ tanh(1 - lat/p50) * weight ] - total_pods * 0.05

    p50: reward_config.json'daki global p50 değerleri kullanılır.
    """
    reward = 0.0
    total_pods = 0

    for svc in SERVICES:
        w   = SERVICE_WEIGHTS.get(svc, 0.0)
        cfg = reward_cfg.get(svc, {})

        if w == 0.0 or not cfg.get('in_reward', True):
            continue

        lat  = raw_metrics[svc]['latency']
        p50  = cfg.get('p50', 1000.0)

        if p50 > 0:
            contribution = np.tanh(1.0 - lat / p50) * w
            reward += contribution

        total_pods += raw_metrics[svc]['num_pods']

    # Pod cezası
    reward -= total_pods * 0.005

    return float(reward)


# ─── Policy ──────────────────────────────────────────────────────────────────

def select_actions_with_info(agents: dict, obs: dict,
                              current_pods: dict, temperature: float) -> tuple:
    """
    Aksiyon seç — log_prob ve value da döndür (PPO update için gerekli).
    Döndürür: (actions, log_probs, values)
    """
    actions   = {}
    log_probs = {}
    values    = {}

    for svc in SERVICES:
        obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)

        logits = agents[svc].actor(obs_t).squeeze(0)
        value  = agents[svc].critic(obs_t).squeeze().item()

        # Pod maskeleme
        pods   = current_pods.get(svc, 2)
        bounds = POD_BOUNDS[svc]
        if pods <= bounds['min']:
            logits[0] = -1e9
        if pods >= bounds['max']:
            logits[2] = -1e9

        # Temperature sampling
        probs    = F.softmax(logits / temperature, dim=-1)
        dist     = torch.distributions.Categorical(probs=probs)
        action_t = dist.sample()

        actions[svc]   = int(action_t.item())
        log_probs[svc] = float(dist.log_prob(action_t).item())
        values[svc]    = value

    return actions, log_probs, values


# ─── kubectl Scale ───────────────────────────────────────────────────────────

def apply_scaling(actions: dict, current_pods: dict,
                  namespace: str, dry_run: bool = True) -> dict:
    new_pods = {}
    for svc in SERVICES:
        delta  = ACTION_MAP[actions[svc]]
        target = int(np.clip(
            current_pods.get(svc, 2) + delta,
            POD_BOUNDS[svc]['min'], POD_BOUNDS[svc]['max']
        ))
        new_pods[svc] = target
        label = ACTION_LABELS[actions[svc]]

        if dry_run:
            log.info(f"  [DRY] {svc:<12} {label:<12} "
                     f"{current_pods.get(svc,2)} → {target}")
        else:
            if target != current_pods.get(svc, 2):
                cmd = ['kubectl', 'scale', 'deployment', DEPLOYMENTS[svc],
                       f'--replicas={target}', '-n', namespace]
                try:
                    r = subprocess.run(cmd, capture_output=True,
                                       text=True, timeout=15)
                    if r.returncode == 0:
                        log.info(f"  ✓ {svc:<12} {label:<12} "
                                 f"{current_pods.get(svc,2)} → {target}")
                    else:
                        log.error(f"  ✗ {svc:<12} {r.stderr.strip()}")
                except Exception as e:
                    log.error(f"  ✗ {svc:<12} {e}")
            else:
                log.info(f"  ○ {svc:<12} {label:<12} "
                         f"{current_pods.get(svc,2)} (değişmedi)")
    return new_pods


# ─── PPO Online Update ────────────────────────────────────────────────────────

def online_update(agents: dict, buffer: list, entropy_coef: float):
    """
    Buffer'daki geçişlerle her ajan için PPO update yap.
    Buffer format: list of {svc: {obs, action, log_prob, reward, value, done}}
    """
    if len(buffer) == 0:
        return

    # Her ajan için ayrı update
    metrics_all = defaultdict(list)

    for svc in SERVICES:
        # Buffer'dan bu servisin verilerini çek
        obs_list      = [t[svc]['obs']      for t in buffer]
        action_list   = [t[svc]['action']   for t in buffer]
        log_prob_list = [t[svc]['log_prob'] for t in buffer]
        reward_list   = [t[svc]['reward']   for t in buffer]
        value_list    = [t[svc]['value']    for t in buffer]
        done_list     = [t[svc]['done']     for t in buffer]

        # Buffer'ı agent'ın buffer'ına yükle
        agents[svc].buffer.clear()
        for i in range(len(obs_list)):
            agents[svc].buffer.push(
                obs=obs_list[i],
                action=action_list[i],
                log_prob=log_prob_list[i],
                reward=reward_list[i],
                done=done_list[i],
                value=value_list[i],
            )

        # PPO update
        m = agents[svc].update(
            last_obs=obs_list[-1],
            entropy_coef=entropy_coef,
        )
        agents[svc].buffer.clear()

        if m:
            metrics_all['actor_loss'].append(m.get('actor_loss', 0))
            metrics_all['entropy'].append(m.get('entropy', 0))

    avg_actor = np.mean(metrics_all['actor_loss']) if metrics_all['actor_loss'] else 0
    avg_ent   = np.mean(metrics_all['entropy'])    if metrics_all['entropy']    else 0
    log.info(f"  PPO Update — ActorLoss: {avg_actor:.4f} | Entropy: {avg_ent:.4f}")


# ─── Ana Döngü ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG['ONLINE_CKPT_DIR'], exist_ok=True)

    log.info("=" * 65)
    log.info("RL Scaler Online — IPPO v5 Fine-Tuning")
    log.info("=" * 65)
    log.info(f"  Namespace        : {CONFIG['NAMESPACE']}")
    log.info(f"  Prometheus       : {CONFIG['PROMETHEUS_URL']}")
    log.info(f"  Checkpoint       : {CONFIG['CHECKPOINT']}")
    log.info(f"  Step interval    : {CONFIG['STEP_INTERVAL']}s")
    log.info(f"  Temperature      : {CONFIG['TEMPERATURE']}")
    log.info(f"  Update interval  : her {CONFIG['UPDATE_INTERVAL']} adımda")
    log.info(f"  Save interval    : her {CONFIG['SAVE_INTERVAL']} adımda")
    log.info(f"  DRY RUN          : {CONFIG['DRY_RUN']}")
    log.info("=" * 65)

    if CONFIG['DRY_RUN']:
        log.info("  ⚠ DRY RUN aktif — kubectl çalışmayacak")
        log.info("  Test ettikten sonra DRY_RUN=False yapın")
        log.info("=" * 65)

    agents     = load_agents(CONFIG['CHECKPOINT'])
    scaler     = load_scaler(CONFIG['SCALER_PATH'])
    reward_cfg = load_reward_cfg(CONFIG['REWARD_CFG'])

    buffer        = []      # online transitions
    step          = 0
    total_reward  = 0.0
    reward_history = []
    n_steps       = CONFIG['N_STEPS']

    # İlk state
    current_metrics = collect_metrics(
        CONFIG['PROMETHEUS_URL'], CONFIG['NAMESPACE'], CONFIG['METRIC_WINDOW']
    )
    current_state = build_global_state(current_metrics, scaler)
    current_obs   = build_obs(current_state)
    current_pods  = {svc: int(current_metrics[svc]['num_pods']) for svc in SERVICES}

    while True:
        step += 1
        if n_steps and step > n_steps:
            log.info(f"  {n_steps} adım tamamlandı.")
            break

        log.info(f"\n{'─'*65}")
        log.info(f"  Adım {step} | {datetime.now().strftime('%H:%M:%S')}")
        log.info(f"{'─'*65}")

        # 1. Aksiyon seç
        actions, log_probs, values = select_actions_with_info(
            agents, current_obs, current_pods, CONFIG['TEMPERATURE']
        )

        log.info("  Aksiyonlar:")
        for svc in SERVICES:
            a      = actions[svc]
            target = int(np.clip(
                current_pods[svc] + ACTION_MAP[a],
                POD_BOUNDS[svc]['min'], POD_BOUNDS[svc]['max']
            ))
            log.info(f"    {svc:<12} → {ACTION_LABELS[a]:<12} "
                     f"pods: {current_pods[svc]} → {target} "
                     f"(güven: {np.exp(log_probs[svc]):.3f})")

        # 2. Uygula
        new_pods = apply_scaling(
            actions, current_pods, CONFIG['NAMESPACE'], CONFIG['DRY_RUN']
        )

        # 3. Bekle — metrikler stabilize olsun
        log.info(f"\n  {CONFIG['STEP_INTERVAL']}s bekleniyor...")
        time.sleep(CONFIG['STEP_INTERVAL'])

        # 4. Yeni state ve reward
        next_metrics = collect_metrics(
            CONFIG['PROMETHEUS_URL'], CONFIG['NAMESPACE'], CONFIG['METRIC_WINDOW']
        )
        next_state = build_global_state(next_metrics, scaler)
        next_obs   = build_obs(next_state)
        next_pods  = {svc: int(next_metrics[svc]['num_pods']) for svc in SERVICES}

        reward = compute_reward(next_metrics, reward_cfg, scaler,
                                current_pods, next_pods)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        total_reward  += reward
        reward_history.append(reward)

        log.info(f"\n  Metrikler (aksiyon sonrası):")
        log.info(f"  {'Servis':<12} {'Pods':>5} {'CPU':>8} {'Mem(MB)':>9} "
                 f"{'RPS':>8} {'Lat(ms)':>9}")
        log.info("  " + "-" * 55)
        for svc in SERVICES:
            m = next_metrics[svc]
            log.info(f"  {svc:<12} {m['num_pods']:>5.0f} "
                     f"{m['cpu_usage']:>8.3f} "
                     f"{m['mem_usage']:>9.1f} "
                     f"{m['request_rate']:>8.2f} "
                     f"{m['latency']:>9.1f}")

        log.info(f"\n  Reward: {reward:.4f} | "
                 f"Toplam: {total_reward:.4f} | "
                 f"Son 10 ort: {np.mean(reward_history[-10:]):.4f}")

        # 5. Buffer'a ekle — nan/inf temizle
        transition = {}
        for svc in SERVICES:
            clean_obs      = np.nan_to_num(current_obs[svc], nan=0.0, posinf=1.0, neginf=0.0)
            clean_next_obs = np.nan_to_num(next_obs[svc],    nan=0.0, posinf=1.0, neginf=0.0)
            transition[svc] = {
                'obs':      clean_obs,
                'action':   actions[svc],
                'log_prob': log_probs[svc],
                'reward':   float(reward),
                'value':    float(values[svc]),
                'done':     False,
            }
        buffer.append(transition)
        # next_obs'u da temizle sonraki adım için
        next_obs = {svc: np.nan_to_num(next_obs[svc], nan=0.0, posinf=1.0, neginf=0.0)
                    for svc in SERVICES}

        # 6. PPO Update
        if step % CONFIG['UPDATE_INTERVAL'] == 0:
            log.info(f"\n  PPO Update ({len(buffer)} transition)...")
            online_update(agents, buffer, CONFIG['ENTROPY_COEF'])
            buffer.clear()

        # 7. Checkpoint + XAI Hook
        if step % CONFIG['SAVE_INTERVAL'] == 0:
            save_checkpoint(agents, step, total_reward, CONFIG['ONLINE_CKPT_DIR'])

            # ── XAI Pipeline'ı arka planda tetikle ──────────────────────────
            # Popen kullanılıyor: scaler'ı bloklamaz, arka planda çalışır
            if CONFIG.get('XAI_ENABLED', True) and not CONFIG['DRY_RUN']:
                ckpt_path  = os.path.abspath(
                    os.path.join(CONFIG['ONLINE_CKPT_DIR'], f'online_step{step}.pth'))
                output_dir = os.path.abspath(
                    os.path.join(CONFIG['XAI_OUTPUT_DIR'], f'step{step}'))
                base_dir   = os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__)))
                xai_runner = os.path.abspath(
                    os.path.join(CONFIG['XAI_SCRIPT_DIR'], 'xai_runner.py'))

                xai_cmd = [
                    sys.executable, xai_runner,
                    '--checkpoint', ckpt_path,
                    '--output_dir', output_dir,
                    '--base_dir',   base_dir,
                    '--skip_step4',   # Step4 (LLM) ağır, sadece finalde çalıştır
                ]
                try:
                    # Windows'ta yeni terminal penceresinde açar
                    extra = {}
                    xai_env = os.environ.copy()
                    xai_env['PYTHONUTF8'] = '1'
                    xai_env['PYTHONIOENCODING'] = 'utf-8'
                    if sys.platform == 'win32':
                        extra['creationflags'] = subprocess.CREATE_NEW_CONSOLE
                    subprocess.Popen(xai_cmd, env=xai_env, **extra)
                    log.info(f"  🔬 XAI pipeline başlatıldı (arka plan) → {output_dir}")
                except Exception as e:
                    log.warning(f"  ⚠ XAI pipeline başlatılamadı: {e}")

        # 8. Sonraki adım için state güncelle
        current_metrics = next_metrics
        current_state   = next_state
        current_obs     = next_obs
        current_pods    = next_pods


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info("\n  Durduruldu. Son checkpoint kaydediliyor...")
        # Kalan buffer'ı güncelle
        log.info("  Tamamlandı.")