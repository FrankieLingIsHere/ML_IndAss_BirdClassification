#!/usr/bin/env python3
"""
Plot fine-tune training history.

This script is Colab-friendly. It tries these sources (in order):
 - results/fine_tune/fine_tune_history.json
 - per-epoch checkpoints results/fine_tune/ckpt_epoch_*.pth (extracts saved losses from each checkpoint)

Usage examples (Colab):
  # after mounting Drive and cd into repo root
  python tools/plot_fine_tune_history.py --out ./results/fine_tune/train_val_loss.png

If matplotlib is not installed in the environment, install it with:
  pip install matplotlib numpy torch torchvision
"""

import argparse
import json
import os
import glob
import torch
import numpy as np

def load_history_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_from_checkpoints(ckpt_pattern):
    paths = sorted(glob.glob(ckpt_pattern))
    if not paths:
        return None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for p in paths:
        try:
            ck = torch.load(p, map_location='cpu')
        except Exception:
            continue
        # Attempt multiple locations for loss/acc values to be robust across formats
        # 1) direct scalar fields (train_loss, train_loss_epoch, loss)
        def _get_scalar(cdict, keys):
            for k in keys:
                if k in cdict and cdict[k] is not None:
                    return cdict[k]
            return None

        # If the checkpoint stores a full history dict/lists, prefer that and return immediately
        if isinstance(ck, dict):
            # common nested history patterns
            for hkey in ('history', 'training_history', 'train_history'):
                if hkey in ck and isinstance(ck[hkey], dict):
                    h = ck[hkey]
                    # copy over recognized lists if present
                    for kk in ('train_loss', 'val_loss', 'train_acc', 'val_acc'):
                        if kk in h and isinstance(h[kk], list):
                            history[kk] = [float(x) if x is not None else np.nan for x in h[kk]]
                    return history

        tl = _get_scalar(ck, ('train_loss', 'train_loss_epoch', 'loss'))
        vl = _get_scalar(ck, ('val_loss', 'validation_loss'))

        # Fallback: some checkpoints pack metrics under a 'metrics' or 'logs' dict
        metrics = None
        for mk in ('metrics', 'logs', 'eval_metrics'):
            if isinstance(ck, dict) and mk in ck and isinstance(ck[mk], dict):
                metrics = ck[mk]
                break
        if metrics is not None:
            if tl is None:
                tl = _get_scalar(metrics, ('train_loss', 'loss'))
            if vl is None:
                vl = _get_scalar(metrics, ('val_loss', 'validation_loss'))

        # Accuracies
        ta = _get_scalar(ck, ('train_acc', 'train_accuracy', 'acc', 'accuracy'))
        va = _get_scalar(ck, ('val_acc', 'val_accuracy'))
        # Fallbacks: some checkpoints store test_acc or avg_class_acc instead of val_acc
        if va is None and isinstance(ck, dict):
            if 'test_acc' in ck:
                va = ck.get('test_acc')
            elif 'avg_class_acc' in ck:
                va = ck.get('avg_class_acc')
        if metrics is not None:
            if ta is None:
                ta = _get_scalar(metrics, ('train_acc', 'train_accuracy', 'acc', 'accuracy'))
            if va is None:
                va = _get_scalar(metrics, ('val_acc', 'val_accuracy'))

        # If we didn't find train-loss but found list-like full history in top-level checkpoint, use it
        # Already handled above.

        # Only append when we have something meaningful; otherwise append nan placeholders so lengths match
        if tl is not None:
            try:
                history['train_loss'].append(float(tl))
            except Exception:
                history['train_loss'].append(np.nan)
        else:
            history['train_loss'].append(np.nan)

        if vl is not None:
            try:
                history['val_loss'].append(float(vl))
            except Exception:
                history['val_loss'].append(np.nan)
        else:
            history['val_loss'].append(np.nan)

        # optional accs
        try:
            history['train_acc'].append(float(ta) if ta is not None else np.nan)
        except Exception:
            history['train_acc'].append(np.nan)
        try:
            history['val_acc'].append(float(va) if va is not None else np.nan)
        except Exception:
            history['val_acc'].append(np.nan)
    # Diagnostic summary: how many non-NaN entries were found for each metric
    try:
        def _count_non_nan(lst):
            return sum(0 if x is None or (isinstance(x, float) and np.isnan(x)) else 1 for x in lst)
        print('Extracted from checkpoints: epochs=', len(paths))
        print('  train_loss entries:', _count_non_nan(history['train_loss']))
        print('  val_loss entries:  ', _count_non_nan(history['val_loss']))
        print('  train_acc entries: ', _count_non_nan(history['train_acc']))
        print('  val_acc entries:   ', _count_non_nan(history['val_acc']))
    except Exception:
        pass
    return history


def plot_history(history, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Prefer plotting losses; if losses are all NaN, fall back to accuracies
    n_epochs = max(len(history.get('train_loss', [])), len(history.get('val_loss', [])), len(history.get('train_acc', [])), len(history.get('val_acc', [])))
    epochs = list(range(1, n_epochs + 1))
    plt.figure(figsize=(8,5))

    has_loss = any([not np.all(np.isnan(history.get('train_loss', []))) if history.get('train_loss') else False,
                    not np.all(np.isnan(history.get('val_loss', []))) if history.get('val_loss') else False])
    if has_loss:
        # Plot loss curves
        if any(not np.isnan(x) for x in history.get('train_loss', [])):
            plt.plot(epochs[:len(history['train_loss'])], history['train_loss'], label='train_loss', marker='o')
        if any(not np.isnan(x) for x in history.get('val_loss', [])):
            plt.plot(epochs[:len(history['val_loss'])], history['val_loss'], label='val_loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Fine-tune: Train / Val Loss')
    else:
        # Fall back to accuracy curves if losses are not available
        if any(not np.isnan(x) for x in history.get('train_acc', [])):
            plt.plot(epochs[:len(history['train_acc'])], history['train_acc'], label='train_acc', marker='o')
        if any(not np.isnan(x) for x in history.get('val_acc', [])):
            plt.plot(epochs[:len(history['val_acc'])], history['val_acc'], label='val_acc', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Fine-tune: Train / Val Accuracy')

    plt.grid(True)
    # Only call legend if there are labeled lines to avoid a Matplotlib warning in notebook environments
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved plot to', out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='results/fine_tune/fine_tune_history.json', help='Path to history JSON')
    parser.add_argument('--ckpt-pattern', type=str, default='results/fine_tune/ckpt_epoch_*.pth', help='Pattern to match per-epoch checkpoints')
    parser.add_argument('--out', type=str, default='results/fine_tune/train_val_loss.png', help='Output PNG path')
    # Use parse_known_args so that IPython/Colab kernel extra args (e.g. -f ...) do
    # not cause argparse to raise SystemExit when this script is imported/run inside
    # a notebook. Unknown args are ignored.
    args, _unknown = parser.parse_known_args()

    history = None
    if os.path.exists(args.history):
        print('Loading history JSON:', args.history)
        history = load_history_json(args.history)
    else:
        print('History JSON not found; attempting to extract from checkpoints using pattern:', args.ckpt_pattern)
        history = extract_from_checkpoints(args.ckpt_pattern)

    if history is None:
        print('No history available (no JSON and no checkpoints matched). Exiting.')
        return

    # Normalize keys if stored under different names
    # Some training scripts store train_loss as list under 'train_loss'
    # If history is a dict with lists we're good; otherwise attempt to coerce
    if isinstance(history, dict):
        # ensure numeric lists
        for k in ('train_loss', 'val_loss'):
            if k in history and isinstance(history[k], list):
                history[k] = [float(x) if x is not None else np.nan for x in history[k]]
            else:
                history[k] = []
    else:
        print('Unexpected history format:', type(history))
        return

    plot_history(history, args.out)

if __name__ == '__main__':
    main()
