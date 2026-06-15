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

#CONFIG 

CONFIG = {
    'PROMETHEUS_URL' : 'http://localhost:9090',   # accessed via kubectl port-forward
    'NAMESPACE'      : 'robot-shop-rl',           # Robot Shop namespace
    'CHECKPOINT'     : 'checkpoints_v5/ippo_v5_final.pth',
    'SCALER_PATH'    : 'scaler.pkl',
    'REWARD_CFG'     : 'reward_config.json',
    'STEP_INTERVAL'  : 90,      # seconds -- consistent with the metric collection window
    'TEMPERATURE'    : 0.8,     # 0.5: more deterministic, 1.5: more exploratory
    'METRIC_WINDOW'  : '2m',    # Prometheus rate() window
    'N_STEPS'        : None,    # None = infinite loop, int = number of steps to run
    'DRY_RUN'        : True,    # LEAVE True ON FIRST RUN -- kubectl will not run
    'LOG_FILE'       : 'rl_scaler.log',
}

# Constants

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

# Deployment names (for kubectl scale)
DEPLOYMENTS = {
    'cart':      'cart',
    'catalogue': 'catalogue',
    'payment':   'payment',
    'shipping':  'shipping',
    'ratings':   'ratings',
    'user':      'user',
}

# Pod bounds (from merge_and_clean.py, same as marl_env)
POD_BOUNDS = {
    'cart':      {'min': 1, 'max': 8},
    'catalogue': {'min': 1, 'max': 8},
    'payment':   {'min': 2, 'max': 8},  # MySQL constraint
    'shipping':  {'min': 2, 'max': 8},  # MySQL constraint
    'ratings':   {'min': 1, 'max': 8},
    'user':      {'min': 1, 'max': 8},
}

ACTION_MAP = {0: -1, 1: 0, 2: 1}   # scale_down, keep, scale_up
ACTION_LABELS = {0: 'scale_down', 1: 'keep', 2: 'scale_up'}

# Partial observability -- same as marl_env.py
AGENT_NEIGHBORS = {
    'cart':      ['catalogue', 'shipping'],
    'catalogue': ['ratings', 'cart'],
    'payment':   ['shipping'],
    'shipping':  ['cart', 'payment'],
    'ratings':   ['catalogue'],
    'user':      ['cart'],
}

# Logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CONFIG['LOG_FILE']),
    ]
)
log = logging.getLogger(__name__)

# Model Loading

def load_agents(checkpoint_path: str):
    """Load the IPPO v5 agents."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ippo_agent import IPPOAgent

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Obs dimensions (same as marl_env.py)
    obs_dims = {
        'cart':      15,  # cart + catalogue + shipping
        'catalogue': 15,  # catalogue + ratings + cart
        'payment':   10,  # payment + shipping
        'shipping':  15,  # shipping + cart + payment
        'ratings':   10,  # ratings + catalogue
        'user':      10,  # user + cart
    }

    agents = {}
    for svc in SERVICES:
        agent = IPPOAgent(obs_dim=obs_dims[svc], n_actions=3, service=svc)
        agent.actor.load_state_dict(ckpt['agents'][svc]['actor'])
        agent.actor.eval()
        agents[svc] = agent

    log.info(f"Model loaded: {checkpoint_path}")
    return agents


def load_scaler(scaler_path: str):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


# Prometheus Metric Collection

def query_prometheus(url: str, query: str) -> float | None:
    """Run a single Prometheus instant query."""
    try:
        resp = requests.get(
            f'{url}/api/v1/query',
            params={'query': query},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get('data', {}).get('result', [])
        if results:
            return float(results[0]['value'][1])
        return None
    except Exception as e:
        log.warning(f"Prometheus query error: {query[:60]}... -> {e}")
        return None


def get_current_pods(namespace: str, deployment: str) -> int | None:
    """Get the current pod count via kubectl."""
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'deployment', deployment,
             '-n', namespace,
             '-o', 'jsonpath={.spec.replicas}'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
        return None
    except Exception as e:
        log.warning(f"kubectl get pods error ({deployment}): {e}")
        return None


def collect_metrics(prometheus_url: str, namespace: str, window: str) -> dict | None:

    metrics = {}

    for svc in SERVICES:
        svc_metrics = {}

        # num_pods -- from kubectl
        pods = get_current_pods(namespace, DEPLOYMENTS[svc])
        if pods is None:
            log.warning(f"{svc}: could not get pod count, defaulting to 2")
            pods = 2
        svc_metrics['num_pods'] = float(pods)

        # cpu_usage -- container CPU usage (cores)
        cpu_query = (
            f'sum(rate(container_cpu_usage_seconds_total{{'
            f'namespace="{namespace}", '
            f'container="{svc}"'
            f'}}[{window}]))'
        )
        cpu = query_prometheus(prometheus_url, cpu_query)
        svc_metrics['cpu_usage'] = cpu if cpu is not None else 0.0

        # mem_usage -- bytes -> MB
        mem_query = (
            f'sum(container_memory_working_set_bytes{{'
            f'namespace="{namespace}", '
            f'container="{svc}"'
            f'}}) / 1048576'
        )
        mem = query_prometheus(prometheus_url, mem_query)
        svc_metrics['mem_usage'] = mem if mem is not None else 0.0

        # request_rate -- Istio (reporter=destination)
        rps_query = (
            f'sum(rate(istio_requests_total{{'
            f'destination_workload_namespace="{namespace}", '
            f'destination_workload=~"{svc}.*", '
            f'reporter="destination"'
            f'}}[{window}]))'
        )
        rps = query_prometheus(prometheus_url, rps_query)
        svc_metrics['request_rate'] = rps if rps is not None else 0.0

        # latency -- p50 (ms)
        lat_query = (
            f'histogram_quantile(0.50, sum(rate('
            f'istio_request_duration_milliseconds_bucket{{'
            f'destination_workload_namespace="{namespace}", '
            f'destination_workload=~"{svc}.*", '
            f'reporter="destination"'
            f'}}[{window}])) by (le))'
        )
        lat = query_prometheus(prometheus_url, lat_query)
        svc_metrics['latency'] = lat if lat is not None else 0.0

        metrics[svc] = svc_metrics
        log.debug(
            f"  {svc:<12} pods={svc_metrics['num_pods']:.0f} "
            f"cpu={svc_metrics['cpu_usage']:.3f} "
            f"mem={svc_metrics['mem_usage']:.1f}MB "
            f"rps={svc_metrics['request_rate']:.2f} "
            f"lat={svc_metrics['latency']:.1f}ms"
        )

    return metrics


# State Construction 

def build_global_state(raw_metrics: dict, scaler) -> np.ndarray:

    row = []
    for svc in SERVICES:
        for metric in METRICS:
            row.append(raw_metrics[svc][metric])

    row_arr = np.array(row, dtype=np.float32).reshape(1, -1)

    # The scaler expects as many columns as it was fit on
    # SAR has 30 columns (6 services x 5 metrics)
    try:
        normalized = scaler.transform(row_arr)[0]
    except Exception as e:
        log.warning(f"Scaler transform error: {e} -- using raw values")
        # Fallback: manual [0,1] clip
        normalized = np.clip(row_arr[0] / (row_arr[0].max() + 1e-8), 0, 1)

    return normalized.astype(np.float32)


def build_obs(global_state: np.ndarray) -> dict:
    """
    Build a partial observation for each agent from the global state.
    Same logic as AGENT_NEIGHBORS in marl_env.py.
    """
    service_idx = {svc: i * len(METRICS) for i, svc in enumerate(SERVICES)}
    obs = {}

    for svc in SERVICES:
        svc_obs = []
        # Own metrics
        start = service_idx[svc]
        svc_obs.extend(global_state[start: start + len(METRICS)])
        # Neighbor metrics
        for neighbor in AGENT_NEIGHBORS[svc]:
            n_start = service_idx[neighbor]
            svc_obs.extend(global_state[n_start: n_start + len(METRICS)])
        obs[svc] = np.array(svc_obs, dtype=np.float32)

    return obs


# Policy — Temperature Sampling + Pod Masking

def select_actions(agents: dict, obs: dict, current_pods: dict,
                   temperature: float = 0.8) -> dict:
    """
    Select an action for each service.
    - Temperature sampling: prevents collapsed policy
    - Pod masking: blocks out-of-bounds actions
    """
    actions = {}

    for svc in SERVICES:
        obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = agents[svc].actor(obs_t).squeeze(0)  # [3]

        # Pod masking -- set out-of-bounds action to -inf
        pods = current_pods.get(svc, 2)
        bounds = POD_BOUNDS[svc]

        if pods <= bounds['min']:
            logits[0] = -1e9   # block scale_down
        if pods >= bounds['max']:
            logits[2] = -1e9   # block scale_up

        # Temperature sampling
        probs = F.softmax(logits / temperature, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        actions[svc] = action

    return actions


#  kubectl Scale 

def apply_scaling(actions: dict, current_pods: dict,
                  namespace: str, dry_run: bool = True) -> dict:
    """
    Apply the actions -- run kubectl scale.
    dry_run=True: only logs, does not run kubectl.
    Returns: {svc: new_pod_count}
    """
    new_pods = {}

    for svc in SERVICES:
        action   = actions[svc]
        delta    = ACTION_MAP[action]
        cur_pods = current_pods.get(svc, 2)
        bounds   = POD_BOUNDS[svc]

        target = int(np.clip(cur_pods + delta, bounds['min'], bounds['max']))
        new_pods[svc] = target

        label = ACTION_LABELS[action]

        if dry_run:
            log.info(
                f"  [DRY RUN] {svc:<12} {label:<12} "
                f"{cur_pods} -> {target} pod"
            )
        else:
            if target != cur_pods:
                cmd = [
                    'kubectl', 'scale', 'deployment',
                    DEPLOYMENTS[svc],
                    f'--replicas={target}',
                    '-n', namespace
                ]
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=15
                    )
                    if result.returncode == 0:
                        log.info(
                            f"  OK {svc:<12} {label:<12} "
                            f"{cur_pods} -> {target} pod"
                        )
                    else:
                        log.error(
                            f"  FAIL {svc:<12} kubectl error: {result.stderr.strip()}"
                        )
                except Exception as e:
                    log.error(f"  FAIL {svc:<12} subprocess error: {e}")
            else:
                log.info(
                    f"  -- {svc:<12} {label:<12} "
                    f"{cur_pods} pod (unchanged)"
                )

    return new_pods


#  Main Loop

def main():
    log.info("=" * 65)
    log.info("RL Scaler v5 — IPPO Entropy Floor")
    log.info("=" * 65)
    log.info(f"  Namespace      : {CONFIG['NAMESPACE']}")
    log.info(f"  Prometheus     : {CONFIG['PROMETHEUS_URL']}")
    log.info(f"  Checkpoint     : {CONFIG['CHECKPOINT']}")
    log.info(f"  Step interval  : {CONFIG['STEP_INTERVAL']}s")
    log.info(f"  Temperature    : {CONFIG['TEMPERATURE']}")
    log.info(f"  DRY RUN        : {CONFIG['DRY_RUN']}")
    log.info("=" * 65)

    if CONFIG['DRY_RUN']:
        log.info("  WARNING DRY RUN active -- kubectl will not run")
        log.info("  For real scaling, set CONFIG['DRY_RUN'] = False")
        log.info("=" * 65)

    # Load model and scaler
    agents = load_agents(CONFIG['CHECKPOINT'])
    scaler = load_scaler(CONFIG['SCALER_PATH'])

    step = 0
    n_steps = CONFIG['N_STEPS']

    while True:
        step += 1
        if n_steps and step > n_steps:
            log.info(f"  {n_steps} steps completed, exiting.")
            break

        log.info(f"\n{'-'*65}")
        log.info(f"  Step {step} | {datetime.now().strftime('%H:%M:%S')}")
        log.info(f"{'-'*65}")

        # 1. Get current pod counts
        current_pods = {}
        for svc in SERVICES:
            pods = get_current_pods(CONFIG['NAMESPACE'], DEPLOYMENTS[svc])
            current_pods[svc] = pods if pods is not None else POD_BOUNDS[svc]['min']

        log.info("  Current pod counts:")
        log.info("  " + "  ".join(
            f"{svc}={current_pods[svc]}" for svc in SERVICES
        ))

        # 2. Collect metrics from Prometheus
        log.info(f"  Collecting metrics ({CONFIG['METRIC_WINDOW']} window)...")
        raw_metrics = collect_metrics(
            CONFIG['PROMETHEUS_URL'],
            CONFIG['NAMESPACE'],
            CONFIG['METRIC_WINDOW']
        )

        if raw_metrics is None:
            log.error("  Could not collect metrics, skipping step.")
            time.sleep(CONFIG['STEP_INTERVAL'])
            continue

        # Log the metrics
        log.info(f"  {'Service':<12} {'Pods':>5} {'CPU':>8} {'Mem(MB)':>9} {'RPS':>8} {'Lat(ms)':>9}")
        log.info("  " + "-" * 55)
        for svc in SERVICES:
            m = raw_metrics[svc]
            log.info(
                f"  {svc:<12} {m['num_pods']:>5.0f} "
                f"{m['cpu_usage']:>8.3f} "
                f"{m['mem_usage']:>9.1f} "
                f"{m['request_rate']:>8.2f} "
                f"{m['latency']:>9.1f}"
            )

        # 3. Build state
        global_state = build_global_state(raw_metrics, scaler)
        obs = build_obs(global_state)

        # 4. Run policy
        actions = select_actions(
            agents, obs, current_pods,
            temperature=CONFIG['TEMPERATURE']
        )

        log.info(f"\n  Actions:")
        for svc in SERVICES:
            a = actions[svc]
            log.info(
                f"  {svc:<12} -> {ACTION_LABELS[a]:<12} "
                f"(pods: {current_pods[svc]} -> "
                f"{int(np.clip(current_pods[svc] + ACTION_MAP[a], POD_BOUNDS[svc]['min'], POD_BOUNDS[svc]['max']))})"
            )

        # 5. Apply scaling
        log.info(f"\n  Applying scaling (dry_run={CONFIG['DRY_RUN']}):")
        new_pods = apply_scaling(
            actions, current_pods,
            CONFIG['NAMESPACE'],
            dry_run=CONFIG['DRY_RUN']
        )

        # 6. Wait
        log.info(f"\n  Waiting {CONFIG['STEP_INTERVAL']}s...")
        time.sleep(CONFIG['STEP_INTERVAL'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info("\n  Stopped by user.")
