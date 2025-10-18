"""
Wrapper to run the evaluation pipeline in sequence and collect outputs under a timestamped folder in ./results.

Sequence:
 - evaluate_final_model.py
 - evaluate_with_tta.py
 - evaluate_ensemble.py
 - export_misclassified.py

Each script will be invoked with --out-dir pointing into the timestamped run folder's subdirectories.
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

SCRIPTS = [
    ('evaluate_final_model.py', 'eval_single'),
    ('evaluate_with_tta.py', 'eval_tta'),
    ('evaluate_ensemble.py', 'ensemble'),
    ('export_misclassified.py', 'misclassified'),
]

def main():
    parser = argparse.ArgumentParser(description='Run full evaluation pipeline and collect results under ./results/<timestamp>')
    parser.add_argument('--prefix', type=str, default='', help='Optional prefix for the run folder')
    parser.add_argument('--python', type=str, default=sys.executable, help='Python executable to use')
    parser.add_argument('--model', type=str, default=None, help='Path to a checkpoint to evaluate; forwarded to evaluate_final_model.py')
    parser.add_argument('--no-tta', action='store_true', help='Pass --no-tta to TTA/ensemble/export steps')
    parser.add_argument('--copy-mis', type=int, default=5, help='Number of misclassified images to copy per class')
    parser.add_argument('--models', nargs='*', help='Optional explicit list of model checkpoints to pass to ensemble/export scripts')
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('results', f"{args.prefix + '_' if args.prefix else ''}{ts}")
    os.makedirs(run_dir, exist_ok=True)

    print('Run dir:', run_dir)

    # Call each script in sequence
    for script, sub in SCRIPTS:
        out_subdir = os.path.join(run_dir, sub)
        cmd = [args.python, script, '--out-dir', out_subdir]
        # Forward model to the single-eval script if provided
        # Forward model to scripts that accept it so the pipeline uses the provided checkpoint
        if args.model:
            if script == 'evaluate_final_model.py' or script == 'evaluate_with_tta.py':
                cmd += ['--model', args.model]
            # evaluate_ensemble.py and export_misclassified.py accept --models (plural)
            if script in ('evaluate_ensemble.py', 'export_misclassified.py'):
                cmd += ['--models', args.model]
        if script in ('evaluate_ensemble.py', 'export_misclassified.py') and args.models:
            cmd += ['--models'] + args.models
        if args.no_tta and script in ('evaluate_with_tta.py', 'evaluate_ensemble.py', 'export_misclassified.py'):
            cmd.append('--no-tta')
        if script == 'export_misclassified.py':
            cmd += ['--copy', str(args.copy_mis)]
        print('\nRunning:', ' '.join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Script {script} failed with exit {e.returncode}. Continuing to next step.")

    print('\nEvaluation pipeline completed. Results collected under', run_dir)

if __name__ == '__main__':
    main()
