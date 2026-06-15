import argparse
import logging
import random
import subprocess
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration

NAMESPACE = "robot-shop"

SERVICES = [
    "cart",
    "catalogue",
    "payment",
    "shipping",
    "user",
    "ratings",
    "dispatch",
]

SERVICE_BOUNDS: dict[str, tuple[int, int]] = {
    "cart":      (1, 8),
    "catalogue": (1, 8),
    "payment":   (2, 8),
    "shipping":  (2, 8),
    "user":      (1, 8),
    "ratings":   (1, 8),
    "dispatch":  (1, 4),
}

STRATEGIES = ["random", "low_load", "high_load", "mixed"]

# Kubernetes helpers

def kubectl_scale(service: str, replicas: int) -> bool:
    cmd = [
        "kubectl", "scale", "deployment", service,
        f"--replicas={replicas}",
        "-n", NAMESPACE,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def reset_all(replicas: int = 2) -> None:
    logger.info("Resetting all services to %d replicas.", replicas)
    for svc in SERVICES:
        kubectl_scale(svc, replicas)
    logger.info("Reset complete.")


# Replica selection

def select_replicas(service: str, strategy: str) -> int:
    lo, hi = SERVICE_BOUNDS[service]

    if strategy == "random":
        replicas = random.randint(lo, hi)

    elif strategy == "low_load":
        replicas = random.randint(lo, min(lo + 1, hi))

    elif strategy == "high_load":
        high_lo  = max(lo, hi - 2)
        replicas = random.randint(high_lo, hi)

    elif strategy == "mixed":
        if service in ("cart", "catalogue", "user"):
            replicas = random.randint(max(lo, 4), hi)
        elif service in ("payment", "shipping"):
            replicas = random.randint(2, min(4, hi))
        else:
            replicas = random.randint(lo, min(lo + 2, hi))

    else:
        replicas = random.randint(lo, hi)

    return max(lo, min(replicas, hi))


# Main exploration loop

def run(duration_minutes: int, interval_seconds: int) -> None:
    logger.info("Starting structured exploration.")
    logger.info(
        "Duration: %d min | Interval: %d s | Namespace: %s",
        duration_minutes, interval_seconds, NAMESPACE,
    )
    expected_steps = (duration_minutes * 60) // interval_seconds
    logger.info("Expected steps: ~%d", expected_steps)

    deadline   = time.time() + duration_minutes * 60
    step_count = 0

    try:
        while time.time() < deadline:
            step_count += 1
            remaining_min = int(deadline - time.time()) // 60
            strategy      = STRATEGIES[step_count % len(STRATEGIES)]

            logger.info(
                "Step %d | strategy: %-10s | remaining: ~%d min",
                step_count, strategy, remaining_min,
            )

            for svc in SERVICES:
                replicas = select_replicas(svc, strategy)
                ok       = kubectl_scale(svc, replicas)
                status   = "scaled" if ok else "ERROR"
                logger.info("  %-12s  %s -> %d pods", svc, status, replicas)

            logger.info(
                "Waiting %d s for metrics to propagate.", interval_seconds
            )
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        reset_all()
        logger.info(
            "Exploration finished after %d step(s). "
            "Run collect_metrics.py to collect the recorded metrics.",
            step_count,
        )


# Entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured pod-scaling exploration for RL data collection."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=720,
        metavar="MINUTES",
        help="Total exploration time in minutes (default: 720 = 12 hours).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=90,
        metavar="SECONDS",
        help=(
            "Seconds between consecutive scaling actions (default: 90). "
            "Aligned with the 2-minute Prometheus rate window."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.duration, args.interval)
