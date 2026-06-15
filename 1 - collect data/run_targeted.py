import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Run plan
@dataclass(frozen=True)
class TargetedRun:
    id:            int
    users:         int
    ramp_up:       int      
    duration_min:  int
    output_file:   str
    scaler_strategy: str    
    note:          str


TARGETED_RUNS: list[TargetedRun] = [
    TargetedRun(
        id=11,
        users=40,
        ramp_up=5,
        duration_min=240,
        output_file="dataset_run11_40user_extremelow.csv",
        scaler_strategy="extreme_low_dominant",
        note="Low load + low pod — cart/catalogue/user 1-2 pods",
    ),
    TargetedRun(
        id=12,
        users=150,
        ramp_up=20,
        duration_min=240,
        output_file="dataset_run12_150user_extremelow.csv",
        scaler_strategy="extreme_low_dominant",
        note="Medium load + low pod — under-provisioning scenario",
    ),
    TargetedRun(
        id=13,
        users=400,
        ramp_up=45,
        duration_min=240,
        output_file="dataset_run13_400user_allstrats.csv",
        scaler_strategy="all",
        note="High load + all strategies — capacity boundary enrichment",
    ),
]


# Configuration

NAMESPACE        = "robot-shop"
ROBOT_SHOP_HOST  = "http://localhost:8080"
LOCUST_FILE      = "robot_shop.py"
SCALER_SCRIPT    = "random_scaler_v2.py"   # using v2
COLLECT_SCRIPT   = "collect_metrics.py"
POST_RUN_SETTLE  = 60

SERVICES = [
    "cart", "catalogue", "payment", "shipping",
    "user", "ratings", "dispatch",
]


# Kubernetes helpers

def reset_pods(replicas: int = 2) -> None:
    logger.info("Resetting all services to %d replicas.", replicas)
    for svc in SERVICES:
        subprocess.run(
            ["kubectl", "scale", "deployment", svc,
             f"--replicas={replicas}", "-n", NAMESPACE],
            capture_output=True,
        )


def wait_for_pods(timeout_seconds: int = 180) -> bool:
    logger.info("Waiting for pods to reach 2/2 Running state.")
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "--no-headers"],
            capture_output=True, text=True,
        )
        lines     = result.stdout.strip().splitlines()
        not_ready = [l for l in lines if "2/2" not in l and l.strip()]

        if not not_ready:
            logger.info("All pods are ready.")
            return True

        logger.info("%d pod(s) not yet ready. Retrying in 10s.", len(not_ready))
        time.sleep(10)

    logger.warning("Pod readiness timeout. Proceeding anyway.")
    return False


def upload_to_gdrive(filepath: str) -> bool:
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


# Single run execution

def execute_run(run: TargetedRun, dry_run: bool = False) -> bool:
    logger.info(
        "--- Run %d | %d users | strategy: %s | duration: %d min | output: %s ---",
        run.id, run.users, run.scaler_strategy, run.duration_min, run.output_file,
    )
    logger.info("    Note: %s", run.note)

    scaler_cmd = [
        sys.executable, SCALER_SCRIPT,
        "--duration",  str(run.duration_min),
        "--interval",  "90",
        "--strategy",  run.scaler_strategy,
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

    logger.info("Launching random_scaler_v2.py (strategy: %s).", run.scaler_strategy)
    scaler_proc = subprocess.Popen(scaler_cmd)

    logger.info("Launching Locust (%d users).", run.users)
    locust_proc = subprocess.Popen(locust_cmd)

    locust_proc.wait()
    logger.info("Locust finished.")

    if scaler_proc.poll() is None:
        logger.info("Terminating random_scaler_v2.py.")
        scaler_proc.terminate()
        scaler_proc.wait()
    logger.info("Scaler finished.")

    logger.info("Waiting %ds for Prometheus to finalise metrics.", POST_RUN_SETTLE)
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
        description="Targeted data collection — Run 11, 12, 13."
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        metavar="ID",
        default=None,
        help="Run IDs to execute (e.g.: --runs 11 12). Leave empty for all.",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=180,
        metavar="SECONDS",
        help="Wait time between runs in seconds (default: 180).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_ids = args.runs or [r.id for r in TARGETED_RUNS]
    plan         = [r for r in TARGETED_RUNS if r.id in selected_ids]

    if not plan:
        valid = [r.id for r in TARGETED_RUNS]
        logger.error("No valid run ID. Available: %s", valid)
        sys.exit(1)

    total_min = sum(r.duration_min for r in plan) + (
        (len(plan) - 1) * args.cooldown // 60
    )
    eta = datetime.now() + timedelta(minutes=total_min)

    print()
    print("=" * 75)
    print("  Robot Shop RL — Targeted Data Collection (Run 11-13)")
    print("=" * 75)
    print(f"  {'Run':<5}  {'Users':>6}  {'Strategy':<25}  {'Duration':>8}  Note")
    print("  " + "-" * 72)
    for r in plan:
        print(
            f"  {r.id:<5}  {r.users:>6}  {r.scaler_strategy:<25}  "
            f"{r.duration_min:>6}min  {r.note}"
        )
    print("  " + "-" * 72)
    print(f"  Estimated total : ~{total_min // 60} hours {total_min % 60} minutes")
    print(f"  Estimated finish: {eta.strftime('%d %b %H:%M')}")
    print()
    print("  Goal: collect 1-2 pod data for cart, catalogue, user.")
    print("  In the quality report, the min pod for these services")
    print("  showing as 1 or 2 is an indicator of success.")
    print("=" * 75)
    print()

    if not args.dry_run:
        try:
            input("Press ENTER to start, Ctrl+C to cancel.\n")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    wall_start = time.time()
    failures: list[int] = []

    for i, run in enumerate(plan):
        success = execute_run(run, dry_run=args.dry_run)
        if not success:
            failures.append(run.id)

        if i < len(plan) - 1 and not args.dry_run:
            logger.info("Cool-down: %ds.", args.cooldown)
            reset_pods(2)
            time.sleep(args.cooldown)

    elapsed = int(time.time() - wall_start)
    print()
    print("=" * 75)
    print("  Pipeline completed.")
    print(f"  Elapsed time: {elapsed // 3600} hours {(elapsed % 3600) // 60} minutes")
    if failures:
        print(f"  Failed runs: {failures}")
    else:
        print("  All runs completed successfully.")
    print()
    print("  Next step: add the new CSVs to merge_and_clean.py,")
    print("  then retrain the DT and MARL.")
    print("=" * 75)
    print()

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()