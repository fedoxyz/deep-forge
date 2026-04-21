"""
Training callbacks — TensorBoard (with flush!), JSON log, progress, early stopping.
"""

import os
import time
import json
from typing import Any, Dict, Optional
from datetime import datetime

from backend.core.base import BaseCallback


class TensorBoardCallback(BaseCallback):
    """Log to TensorBoard. Flushes every epoch so data actually shows up."""

    def __init__(self, log_dir: str = '/data/logs/tensorboard', run_name: Optional[str] = None, initial_step=0):
        if run_name:
            self.log_dir = os.path.join(log_dir, run_name)
        else:
            self.log_dir = os.path.join(log_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.writer = None
        self.initial_step = initial_step

    def _w(self):
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            print(f"[TensorBoard] Writing to {self.log_dir}")
        return self.writer

    def on_train_start(self, state):
        w = self._w()
        if 'config' in state:
            w.add_text('config', json.dumps(state.get('config', {}), indent=2, default=str), 0)
        w.flush()

    def on_step_end(self, step, loss, state):
        w = self._w()
        w.add_scalar('train/step_loss', loss, step)
        if 'smoothed_loss' in state:
            w.add_scalar('train/smoothed_loss', state['smoothed_loss'], step)
        if 'lr' in state:
            w.add_scalar('train/learning_rate', state['lr'], step)
        if 'running_accuracy' in state:
            w.add_scalar('train/running_accuracy', state['running_accuracy'], step)
        # Flush every 50 steps
        if step % 50 == 0:
            w.flush()

    def on_epoch_end(self, epoch, state):
        w = self._w()
        if 'epoch_loss' in state:
            w.add_scalar('train/epoch_loss', state['epoch_loss'], epoch)
        if 'train_accuracy' in state:
            w.add_scalar('train/epoch_accuracy', state['train_accuracy'], epoch)
        if 'val_loss' in state:
            w.add_scalar('val/loss', state['val_loss'], epoch)
        if isinstance(state.get('val_metrics'), dict):
            if 'accuracy' in state['val_metrics']:
                w.add_scalar('val/accuracy', state['val_metrics']['accuracy'], epoch)
        # FLUSH so standalone TensorBoard container sees data immediately
        w.flush()

    def on_train_end(self, state):
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None


class JSONLogCallback(BaseCallback):
    """Log to JSON file for the API/frontend."""

    def __init__(self, log_path: str = './logs/training_log.json'):
        self.log_path = log_path
        self.log_data = {
            'steps': [], 'epochs': [], 'config': {},
            'status': 'idle', 'start_time': None, 'end_time': None,
        }
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)

    def _save(self):
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self.log_data, f, indent=2, default=str)
        except Exception:
            pass

    def on_train_start(self, state):
        self.log_data['status'] = 'training'
        self.log_data['start_time'] = datetime.now().isoformat()
        self.log_data['config'] = state.get('config', {})
        self.log_data['run_dir'] = state.get('run_dir', '')
        self._save()

    def on_step_end(self, step, loss, state):
        entry = {'step': step, 'loss': loss, 'timestamp': datetime.now().isoformat()}
        if 'lr' in state:
            entry['lr'] = state['lr']
        if 'smoothed_loss' in state:
            entry['smoothed_loss'] = state['smoothed_loss']
        self.log_data['steps'].append(entry)
        if step % 10 == 0:
            self._save()

    def on_epoch_end(self, epoch, state):
        entry = {'epoch': epoch, 'timestamp': datetime.now().isoformat()}
        for k in ('epoch_loss', 'val_loss', 'train_accuracy'):
            if k in state:
                entry[k] = state[k]
        if isinstance(state.get('val_metrics'), dict):
            entry['val_accuracy'] = state['val_metrics'].get('accuracy')
        self.log_data['epochs'].append(entry)
        self._save()

    def on_train_end(self, state):
        self.log_data['status'] = 'completed'
        self.log_data['end_time'] = datetime.now().isoformat()
        self._save()


class ProgressCallback(BaseCallback):
    def __init__(self, print_every: int = 10):
        self.print_every = print_every
        self.t0 = None; self.et0 = None

    def on_train_start(self, state):
        self.t0 = time.time()
        print(f"\n{'='*60}\nTraining started | mode={state.get('mode','?')} | run_dir={state.get('run_dir','?')}\n{'='*60}")

    def on_epoch_start(self, epoch, state):
        self.et0 = time.time()
        print(f"\n--- Epoch {epoch+1} ---")

    def on_step_end(self, step, loss, state):
        if step > 0 and step % self.print_every == 0:
            e = time.time() - self.t0 if self.t0 else 0
            lr = state.get('lr', 0)
            sm = state.get('smoothed_loss', loss)
            acc = state.get('running_accuracy')
            acc_str = f' | Acc: {acc:.3f}' if acc is not None else ''
            print(f"  Step {step} | Loss: {loss:.5f} (avg: {sm:.5f}) | LR: {lr:.2e}{acc_str} | {e:.0f}s")

    def on_epoch_end(self, epoch, state):
        dt = time.time() - self.et0 if self.et0 else 0
        parts = [f"Epoch {epoch+1} done in {dt:.1f}s"]
        if 'val_loss' in state:
            parts.append(f"val_loss={state['val_loss']:.5f}")
        if isinstance(state.get('val_metrics'), dict) and 'accuracy' in state['val_metrics']:
            parts.append(f"val_acc={state['val_metrics']['accuracy']:.3f}")
        print(' | '.join(parts))

    def on_train_end(self, state):
        t = time.time() - self.t0 if self.t0 else 0
        print(f"\n{'='*60}\nTraining completed in {t:.1f}s | best_val_loss={state.get('best_val_loss','?')}\n{'='*60}\n")


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience; self.min_delta = min_delta
        self.best = float('inf'); self.count = 0; self.should_stop = False

    def on_epoch_end(self, epoch, state):
        vl = state.get('val_loss', float('inf'))
        if vl < self.best - self.min_delta:
            self.best = vl; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
                print(f"[EarlyStopping] Patience exhausted after {self.patience} epochs")
