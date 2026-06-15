import sys
import io
# Windows cp1252 encoding fix -- must be before any print/log
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import argparse
import os
import sys
import subprocess
import logging
from datetime import datetime

# Arguments

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  required=True,
                    help='Full path to the .pth checkpoint file to use')
parser.add_argument('--output_dir',  required=True,
                    help='Folder where all XAI outputs will be written')
parser.add_argument('--base_dir',    default='.',
                    help='Folder containing marl_env.py, scaler.pkl, reward_config.json')
parser.add_argument('--skip_step4',  action='store_true',
                    help='Skip step4 (LLM) for a faster run')
args = parser.parse_args()

CHECKPOINT  = os.path.abspath(args.checkpoint)
OUTPUT_DIR  = os.path.abspath(args.output_dir)
BASE_DIR    = os.path.abspath(args.base_dir)
SKIP_STEP4  = args.skip_step4

# XAIguncel folder -- where this script lives
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging

log_path = os.path.join(OUTPUT_DIR, 'xai_runner.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding='utf-8'),
    ]
)
log = logging.getLogger('xai_runner')


def run_step(script_name: str, extra_args: list) -> bool:
    """Run a step script. Returns True on success."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = [sys.executable, script_path] + extra_args
    log.info(f"  Running: {script_name}")
    log.info(f"  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,   # 10 minutes -- step1 can take a while
            env=env,
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                log.info(f"    {line}")
        if result.returncode != 0:
            log.error(f"  FAIL {script_name} exited with an error (returncode={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-10:]:
                    log.error(f"    {line}")
            return False
        log.info(f"  OK {script_name} completed")
        return True
    except subprocess.TimeoutExpired:
        log.error(f"  FAIL {script_name} timed out (600s)")
        return False
    except Exception as e:
        log.error(f"  FAIL {script_name} could not be run: {e}")
        return False

# Main Flow

def main():
    start_time = datetime.now()

    log.info("=" * 65)
    log.info("XAI Runner — Online Fine-Tuning Pipeline")
    log.info("=" * 65)
    log.info(f"  Checkpoint  : {CHECKPOINT}")
    log.info(f"  Output dir  : {OUTPUT_DIR}")
    log.info(f"  Base dir    : {BASE_DIR}")
    log.info(f"  Skip step4  : {SKIP_STEP4}")
    log.info(f"  Start time  : {start_time.strftime('%H:%M:%S')}")
    log.info("=" * 65)

    # Check whether the checkpoint exists
    if not os.path.exists(CHECKPOINT):
        log.error(f"Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    results = {}

    #  Step 1
    log.info("\n[STEP 1] Behavior Dataset Collection")
    ok = run_step('step1_behavior_dataset.py', [
        '--checkpoint', CHECKPOINT,
        '--output_dir', OUTPUT_DIR,
        '--base_dir',   BASE_DIR,
    ])
    results['step1'] = ok
    if not ok:
        log.error("Step1 failed -- stopping pipeline.")
        sys.exit(1)

    # Step 2 
    log.info("\n[STEP 2] Surrogate Model + Feature Importance")
    ok = run_step('step2_surrogate_shap.py', [
        '--input_dir',  OUTPUT_DIR,
        '--output_dir', OUTPUT_DIR,
    ])
    results['step2'] = ok
    if not ok:
        log.error("Step2 failed -- stopping pipeline.")
        sys.exit(1)

    #  Step 3
    log.info("\n[STEP 3] Rule-based Diagnosis Engine")
    ok = run_step('step3_diagnosis_engine.py', [
        '--input_dir',  OUTPUT_DIR,
        '--output_dir', OUTPUT_DIR,
        '--base_dir',   BASE_DIR,
    ])
    results['step3'] = ok
    if not ok:
        log.warning("Step3 failed -- skipping step4.")
        SKIP_STEP4_local = True
    else:
        SKIP_STEP4_local = SKIP_STEP4

    #  Step 4
    if not SKIP_STEP4_local:
        log.info("\n[STEP 4] Evidence-Guided LLM Explanation")
        ok = run_step('step4_evidence_guided_llm.py', [
            '--input_dir',  OUTPUT_DIR,
            '--output_dir', OUTPUT_DIR,
        ])
        results['step4'] = ok
    else:
        log.info("\n[STEP 4] Skipped (--skip_step4 or step3 error)")
        results['step4'] = None

    #  Summary
    elapsed = (datetime.now() - start_time).seconds
    log.info("\n" + "=" * 65)
    log.info("XAI Runner Complete")
    log.info(f"  Duration : {elapsed}s")
    log.info(f"  Step 1   : {'OK' if results.get('step1') else 'FAIL'}")
    log.info(f"  Step 2   : {'OK' if results.get('step2') else 'FAIL'}")
    log.info(f"  Step 3   : {'OK' if results.get('step3') else 'FAIL'}")
    log.info(f"  Step 4   : {'OK' if results.get('step4') else ('skipped' if results.get('step4') is None else 'FAIL')}")
    log.info(f"  Outputs  : {OUTPUT_DIR}")
    log.info("=" * 65)

if __name__ == '__main__':
    main()
