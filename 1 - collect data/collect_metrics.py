import argparse
import logging
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



# Configuration
PROMETHEUS_URL = "http://localhost:9090/api/v1/query_range"

SERVICES = [
    "cart",
    "catalogue",
    "dispatch",
    "payment",
    "ratings",
    "shipping",
    "user",
    "web",
]

METRIC_NAMES = ["num_pods", "cpu_usage", "mem_usage", "request_rate", "latency"]

# Services expected to carry HTTP traffic, used for quality reporting only.
ACTIVE_SERVICES = ["cart", "catalogue", "payment", "shipping", "ratings", "user"]

# Number of containers per pod in this cluster.
# container_memory_working_set_bytes counts all containers including
# the Istio sidecar (istio-proxy). Each pod has exactly 2 containers:
# the application container and istio-proxy. Dividing the raw count
# by this value yields the true pod count.
CONTAINERS_PER_POD = 2

ISTIO_PROXY = "istio-proxy"

# Prometheus rate window. Must be >= scaler interval (90s) to ensure each
# scaling action has propagated before the next one occurs.
RATE_WINDOW = "2m"


QUERIES: dict[str, str] = {
    "num_pods": (
        'count(count(container_memory_working_set_bytes'
        '{namespace="robot-shop", pod=~"{svc}-.*",'
        ' container!=""}) by (pod))'
    ),
    "cpu_usage": (
        'sum(rate(container_cpu_usage_seconds_total'
        '{namespace="robot-shop", pod=~"{svc}-.*",'
        ' container!="' + ISTIO_PROXY + '", container!=""}[' + RATE_WINDOW + ']))'
    ),
    "mem_usage": (
        'sum(container_memory_working_set_bytes'
        '{namespace="robot-shop", pod=~"{svc}-.*",'
        ' container!="' + ISTIO_PROXY + '", container!=""})'
    ),
    "request_rate": (
        'sum(rate(istio_requests_total'
        '{reporter="destination", destination_workload="{svc}"}[' + RATE_WINDOW + ']))'
    ),
    "latency": (
        'sum(rate(istio_request_duration_milliseconds_sum'
        '{reporter="destination", destination_workload="{svc}"}[' + RATE_WINDOW + '])) / '
        'sum(rate(istio_request_duration_milliseconds_count'
        '{reporter="destination", destination_workload="{svc}"}[' + RATE_WINDOW + ']))'
    ),
}

# Prometheus query

def fetch_range(
    query: str,
    start: int,
    end: int,
    step: str,
    timeout: int = 15,
) -> list[tuple[float, float]]:
    params = {"query": query, "start": start, "end": end, "step": step}
    try:
        resp = requests.get(PROMETHEUS_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success" and data["data"]["result"]:
            return data["data"]["result"][0]["values"]
    except requests.RequestException as exc:
        logger.warning("Prometheus request failed: %s", exc)
    except Exception as exc:
        logger.warning("Unexpected error during metric fetch: %s", exc)
    return []

# Data assembly

def collect(window_minutes: int, step: str) -> pd.DataFrame:

    end_ts   = int(time.time())
    start_ts = end_ts - window_minutes * 60

    logger.info(
        "Querying Prometheus | window: %d min | step: %s | rate window: %s",
        window_minutes, step, RATE_WINDOW,
    )
    logger.info(
        "Time range: %s -> %s",
        datetime.fromtimestamp(start_ts).strftime("%H:%M:%S"),
        datetime.fromtimestamp(end_ts).strftime("%H:%M:%S"),
    )

    rows: dict[int, dict[str, float]] = {}

    for svc in SERVICES:
        logger.info("Fetching metrics for service: %s", svc)
        for metric, template in QUERIES.items():
            query  = template.replace("{svc}", svc)
            values = fetch_range(query, start_ts, end_ts, step)
            col    = f"{svc}_{metric}"
            count  = 0

            for ts_str, val_str in values:
                ts = int(float(ts_str))
                try:
                    val = float(val_str)
                    if not np.isfinite(val):
                        val = 0.0
                except (ValueError, TypeError):
                    val = 0.0

                rows.setdefault(ts, {})[col] = val
                count += 1

            logger.info("  %-15s  %d data points", metric, count)

    ordered_cols = [
        f"{svc}_{m}" for svc in SERVICES for m in METRIC_NAMES
    ]
    records = [
        {"date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"), **vals}
        for ts, vals in rows.items()
    ]
    df = pd.DataFrame(records)
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = (
        df[["date"] + ordered_cols]
        .sort_values("date")
        .fillna(0)
        .reset_index(drop=True)
    )

    # Divide num_pods columns by CONTAINERS_PER_POD to correct for
    # Istio sidecar containers counted by container_memory_working_set_bytes.
    # In this cluster each pod has 2 containers: app + istio-proxy.
    pod_cols = [c for c in df.columns if c.endswith("_num_pods")]
    df[pod_cols] = (df[pod_cols] / CONTAINERS_PER_POD).round().astype(int)

    return df

# Quality report

def quality_report(df: pd.DataFrame) -> None:
    """
    Print a per-service data quality summary.

    Highlights services where the zero-latency fraction exceeds 30%, which
    indicates missing Istio metrics (pod not yet ready, Locust not sending
    traffic to that service).

    Also warns if num_pods values appear unexpectedly high, which would
    suggest that Istio sidecars are still being counted despite the filter.
    """
    print()
    print("=" * 72)
    print("Data Quality Report")
    print("=" * 72)
    print(f"  Total rows   : {len(df)}")
    print(f"  Time range   : {df['date'].iloc[0]}  ->  {df['date'].iloc[-1]}")
    print(f"  Rate window  : {RATE_WINDOW}")
    print()
    print(
        f"  {'Service':<12}  {'Pods (min-max)':<16}  "
        f"{'Max RPS':>8}  {'Zero-lat%':>9}  {'Valid rows':>10}"
    )
    print("  " + "-" * 62)

    for svc in ACTIVE_SERVICES:
        lat_col = f"{svc}_latency"
        rps_col = f"{svc}_request_rate"
        pod_col = f"{svc}_num_pods"

        if lat_col not in df.columns:
            continue

        zero_pct   = (df[lat_col] == 0).mean() * 100
        valid_rows = (df[lat_col] > 0).sum()
        pod_range  = f"{int(df[pod_col].min())}-{int(df[pod_col].max())}"
        rps_max    = df[rps_col].max()
        lat_flag   = " [WARN]" if zero_pct >= 30 else ""

        print(
            f"  {svc:<12}  {pod_range:<16}  "
            f"{rps_max:>8.2f}  {zero_pct:>8.1f}%{lat_flag}"
            f"  {valid_rows:>10}"
        )

        if df[pod_col].max() > 15:
            print(
                f"  {'':12}  WARNING: pod count ({int(df[pod_col].max())}) "
                "may include sidecar containers. Check kube-state-metrics."
            )

    print()
    print("  Quality target: zero-latency fraction < 30% per service.")
    print(
        "  If a service shows 0% request_rate throughout, verify that\n"
        "  Locust is routing traffic to it and the Istio sidecar is\n"
        "  injected in the robot-shop namespace."
    )
    print("=" * 72)
    print()

# Entry point

def parse_args() -> argparse.Namespace:
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Collect Prometheus / Istio metrics for the Robot Shop RL dataset."
        )
    )
    parser.add_argument(
        "--window",
        type=int,
        default=720,
        metavar="MINUTES",
        help="Look-back window in minutes (default: 720 = 12 hours).",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="10s",
        metavar="STEP",
        help="Prometheus resolution step (default: 10s).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Output CSV file path. "
            f"Default: dataset_<window>min_{ts_tag}.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out    = args.out or f"dataset_{args.window}min_{ts_tag}.csv"

    logger.info("collect_metrics.py")
    logger.info("  window      : %d min", args.window)
    logger.info("  step        : %s", args.step)
    logger.info("  rate window : %s", RATE_WINDOW)
    logger.info("  output      : %s", out)
    logger.info(
        "  num_pods fix: sidecar containers (%s) excluded via "
        "kube_pod_container_info filter.",
        ISTIO_PROXY,
    )

    df = collect(args.window, args.step)

    if df.empty:
        logger.error(
            "No data collected. Verify that Prometheus is reachable at %s "
            "and that port-forwarding is active.",
            PROMETHEUS_URL,
        )
        sys.exit(1)

    df.to_csv(out, index=False)
    logger.info(
        "Dataset saved: %s (%d rows, %d columns)", out, len(df), len(df.columns)
    )

    quality_report(df)


if __name__ == "__main__":
    main()
