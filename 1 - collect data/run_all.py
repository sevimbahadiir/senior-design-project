import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Run plan
@dataclass(frozen=True)
class Run:
    id:           int
    users:        int
    ramp_up:      int  
    duration_min: int
    output_file:  str


RUN_PLAN: list[Run] = [
    Run(1,   40,  5,  240, "dataset_run1_40user.csv"),
    Run(2,   80, 10,  240, "dataset_run2_80user.csv"),
    Run(3,  120, 15,  240, "dataset_run3_120user.csv"),
    Run(4,  150, 20,  240, "dataset_run4_150user.csv"),
    Run(5,  200, 25,  240, "dataset_run5_200user.csv"),
    Run(6,  250, 30,  240, "dataset_run6_250user.csv"),   # capacity boundary
    Run(7,  300, 35,  240, "dataset_run7_300user.csv"),   # capacity boundary
    Run(8,  350, 40,  240, "dataset_run8_350user.csv"),   # capacity boundary
    Run(9,  400, 45,  240, "dataset_run9_400user.csv"),   # capacity boundary
    Run(10,  500, 55,  240, "dataset_run10_500user.csv"),   # capacity boundary
]


# Configuration

NAMESPACE        = "robot-shop"
ROBOT_SHOP_HOST  = "http://localhost:8080"
LOCUST_FILE      = "robot_shop.py"
SCALER_SCRIPT    = "random_scaler.py"
COLLECT_SCRIPT   = "collect_metrics.py"

# Seconds to wait after Locust finishes before querying Prometheus.
# Allows the final scrape interval and rate window to complete.
POST_RUN_SETTLE  = 60

SERVICES = [
    "cart", "catalogue", "payment", "shipping",
    "user", "ratings", "dispatch",
]



# Kubernetes helpers

def reset_pods(replicas: int = 2) -> None:
    """Scale all managed services to replicas."""
    logger.info("Resetting all services to %d replicas.", replicas)
    for svc in SERVICES:
        subprocess.run(
            [
                "kubectl", "scale", "deployment", svc,
                f"--replicas={replicas}",
                "-n", NAMESPACE,
            ],
            capture_output=True,
        )


def wait_for_pods(timeout_seconds: int = 180) -> bool:

    logger.info("Waiting for pods to reach 2/2 Running state.")
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "--no-headers"],
            capture_output=True,
            text=True,
        )
        lines     = result.stdout.strip().splitlines()
        not_ready = [l for l in lines if "2/2" not in l and l.strip()]

        if not not_ready:
            logger.info("All pods are ready.")
            return True

        logger.info(
            "%d pod(s) not yet ready. Retrying in 10 s.", len(not_ready)
        )
        time.sleep(10)

    logger.warning(
        "Pod readiness timeout after %d s. Proceeding anyway.", timeout_seconds
    )
    return False


# Single run execution

def upload_to_gdrive(filepath: str) -> bool:
    """Upload a file to Google Drive using rclone."""
    filename = filepath.split("\\")[-1].split("/")[-1]
    logger.info("Uploading %s to gdrive:robot-shop-rl", filename)
    result = subprocess.run(
        ["rclone", "copy", filepath, "gdrive:robot-shop-rl"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("Upload complete: %s", filename)
        return True
    else:
        logger.warning("Upload failed: %s", result.stderr.strip())
        return False


def execute_run(run: Run, dry_run: bool = False) -> bool:
    logger.info(
        "--- Run %d | %d users | ramp-up %d s | duration %d min | output: %s ---",
        run.id, run.users, run.ramp_up, run.duration_min, run.output_file,
    )

    scaler_cmd = [
        sys.executable, SCALER_SCRIPT,
        "--duration", str(run.duration_min),
        "--interval", "90",
    ]
    locust_cmd = [
        "locust", "-f", LOCUST_FILE,
        f"--host={ROBOT_SHOP_HOST}",
        "--headless",
        "-u", str(run.users),
        "-r", str(run.ramp_up),
        "--run-time", f"{run.duration_min}m",
        "--stop-timeout", "30",
    ]
    collect_cmd = [
        sys.executable, COLLECT_SCRIPT,
        "--window", str(run.duration_min),
        "--step",   "10s",
        "--out",    run.output_file,
    ]

    if dry_run:
        logger.info("[DRY-RUN] scaler  : %s", " ".join(scaler_cmd))
        logger.info("[DRY-RUN] locust  : %s", " ".join(locust_cmd))
        logger.info("[DRY-RUN] collect : %s", " ".join(collect_cmd))
        return True

    reset_pods(2)
    wait_for_pods()

    logger.info("Launching random_scaler.py.")
    scaler_proc = subprocess.Popen(scaler_cmd)

    logger.info("Launching Locust.")
    locust_proc = subprocess.Popen(locust_cmd)

    locust_proc.wait()
    logger.info("Locust finished.")

    if scaler_proc.poll() is None:
        logger.info("Terminating random_scaler.py.")
        scaler_proc.terminate()
        scaler_proc.wait()
    logger.info("Scaler finished.")

    logger.info(
        "Waiting %d s for Prometheus to finalise metrics.", POST_RUN_SETTLE
    )
    time.sleep(POST_RUN_SETTLE)

    logger.info("Running collect_metrics.py.")
    result = subprocess.run(collect_cmd)

    if result.returncode != 0:
        logger.error("Metric collection failed for Run %d.", run.id)
        return False

    logger.info("Run %d complete. Dataset: %s", run.id, run.output_file)
    upload_to_gdrive(run.output_file)
    return True


# Entry point

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate all Robot Shop RL data collection runs."
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        metavar="ID",
        default=None,
        help=(
            "Run IDs to execute (e.g. --runs 1 2 3). "
            "Executes all runs if omitted."
        ),
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Cool-down period between runs in seconds (default: 120).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_ids = args.runs or [r.id for r in RUN_PLAN]
    plan         = [r for r in RUN_PLAN if r.id in selected_ids]

    if not plan:
        valid = [r.id for r in RUN_PLAN]
        logger.error("No valid run IDs. Available: %s", valid)
        sys.exit(1)

    total_min = sum(r.duration_min for r in plan) + (
        (len(plan) - 1) * args.cooldown // 60
    )
    eta = datetime.now() + timedelta(minutes=total_min)

    print()
    print("=" * 70)
    print("  Robot Shop RL -- Data Collection Pipeline")
    print("=" * 70)
    print(
        f"  {'Run':<5}  {'Users':>6}  {'Duration':>10}  "
        f"{'Note':<20}  Output"
    )
    print("  " + "-" * 65)
    for r in plan:
        note = "capacity boundary" if r.id in (6, 7) else ""
        print(
            f"  {r.id:<5}  {r.users:>6}  "
            f"{r.duration_min:>8} min  {note:<20}  {r.output_file}"
        )
    print("  " + "-" * 65)
    print(f"  Estimated total : ~{total_min // 60} h {total_min % 60} min")
    print(f"  Estimated end   : {eta.strftime('%d %b %H:%M')}")
    print()
    print(
        "  Note: Runs 6 and 7 are capacity boundary tests. They will be\n"
        "  discarded during data processing if zero-latency > 30%."
    )
    print("=" * 70)
    print()

    if not args.dry_run:
        try:
            input("Press ENTER to start, or Ctrl+C to abort.\n")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    wall_start = time.time()
    failures: list[int] = []

    for i, run in enumerate(plan):
        success = execute_run(run, dry_run=args.dry_run)
        if not success:
            failures.append(run.id)

        if i < len(plan) - 1 and not args.dry_run:
            logger.info(
                "Cool-down: %d s before next run.", args.cooldown
            )
            reset_pods(2)
            time.sleep(args.cooldown)

    elapsed = int(time.time() - wall_start)
    print()
    print("=" * 70)
    print("  Pipeline complete.")
    print(f"  Elapsed : {elapsed // 3600} h {(elapsed % 3600) // 60} min")
    if failures:
        print(f"  Failed runs : {failures}")
    else:
        print("  All runs completed successfully.")
    print("=" * 70)
    print()

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()