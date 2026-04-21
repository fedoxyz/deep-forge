"""
Unified trainer.

Key invariants:
  - self.lora_injectors: Dict[str, LoRAInjector]  (always a dict, never singular)
  - self.component_bundle: ComponentBundle (always present, even for single models)
  - Save/load handles per-component LoRA weights separately
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from backend.core.base import BaseCallback


class UnifiedTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List] = None,
        config: Optional[Dict] = None,
        full_config: Optional[Dict] = None,
        lora_injectors: Optional[Dict[str, Any]] = None,
        component_bundle: Optional[Any] = None,
        mode: str = 'lora',
        run_name: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.config = config or {}
        self.full_config = full_config or {}
        self.mode = mode
        self.component_bundle = component_bundle

        # Always a dict: {component_name: LoRAInjector}. Empty if no LoRA.
        self.lora_injectors = lora_injectors if isinstance(lora_injectors, dict) else {}

        self.training_adapter = kwargs.get('training_adapter', None)

        self.epochs = self.config.get('epochs', 10)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.mixed_precision = self.config.get('mixed_precision', 'none')
        self.gradient_checkpointing = self.config.get('gradient_checkpointing', False)
        self.save_every_n_steps = self.config.get('save_every_n_steps', 0)
        self.save_every_n_epochs = self.config.get('save_every_n_epochs', 0)
        self.eval_every_n_steps = self.config.get('eval_every_n_steps', 0)
        base_output = self.config.get('output_dir', './outputs')
        self.device = torch.device(
            self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        # ── Run directory ──
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        rn = run_name or self.full_config.get('logging', {}).get('run_name') or f'{mode}_{ts}'
        self.run_dir = os.path.join(base_output, rn)
        self.checkpoints_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        if self.full_config:
            with open(os.path.join(self.run_dir, 'resolved_config.json'), 'w') as f:
                json.dump(self.full_config, f, indent=2, default=str)
        if self.full_config.get('model_spec'):
            with open(os.path.join(self.run_dir, 'model_spec.json'), 'w') as f:
                json.dump(self.full_config['model_spec'], f, indent=2, default=str)

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0

        # AMP
        self.scaler = None
        self.amp_dtype = torch.float32
        if self.mixed_precision == 'fp16' and self.device.type == 'cuda':
            self.scaler = GradScaler()
            self.amp_dtype = torch.float16
        elif self.mixed_precision == 'bf16' and self.device.type == 'cuda':
            self.amp_dtype = torch.bfloat16

        # ── Sampling ──
        self.pipeline = kwargs.get('pipeline', None)
        self.sampler = None
        if self.pipeline is not None:
            from backend.core.sampling import TrainingSampler
            self.sampler = TrainingSampler(
                pipeline=self.pipeline,
                config=self.full_config,
                run_dir=self.run_dir,
            )

        self.low_vram = self.full_config.get('pipeline', {}).get('low_vram', False) \
                     or self.full_config.get('training', {}).get('low_vram', False)
        self._offload_device = torch.device('cpu') if self.low_vram else self.device

        # ── Manual control signals ──
        self.stop_event = kwargs.get('stop_event', None)
        self._save_requested = False
        self._sample_requested = False
        self._sample_request_data = None

        # ── Checkpoint rotation ──
        self.keep_last_n = self.config.get('keep_last_n_checkpoints', 5)
        self._checkpoint_history = []  # list of paths, oldest first

        if self.gradient_checkpointing:
            self._enable_grad_ckpt()

        if self.low_vram:
            from backend.core.layer_stream import TrainingLayerStreamer
            
            # Find the trainable DiT component
            if self.component_bundle:
                for comp in self.component_bundle:
                    if comp.trainable and comp.module is not None:
                        # Peel the wrapper to get to actual transformer
                        inner_model = getattr(comp.module, 'model', comp.module)
                        self._layer_streamer = TrainingLayerStreamer(
                            model=inner_model,
                            device=self.device,
                            offload_device=self._offload_device,
                            min_param_bytes=50_000_000,  # stream layers >50MB
                            dtype=self.amp_dtype if self.amp_dtype != torch.float32 else None,
                        )
                        self._layer_streamer.enable()
                        
                        est = self._layer_streamer.estimate_peak_vram() if hasattr(self._layer_streamer, 'estimate_peak_vram') else {}
                        print(f"[LayerStream] Enabled on {type(inner_model).__name__}")
                        if est:
                            print(f"[LayerStream] Largest layer: {est['largest_layer']} "
                                  f"({est['largest_layer_gb']:.2f}GB)")
                            print(f"[LayerStream] Est. peak VRAM: {est['estimated_peak_gb']:.2f}GB")
                        break
            else:
                self._layer_streamer = None
        else:
            self._layer_streamer = None

    # ── Helpers ──
    def _enable_grad_ckpt(self):
        for attr in ('enable_gradient_checkpointing', 'gradient_checkpointing_enable'):
            if hasattr(self.model, attr):
                getattr(self.model, attr)()
                return
    
        if self.component_bundle:
            for comp in self.component_bundle:
                if comp.module is None or not comp.trainable:
                    continue
                # ── NEW: for ZImageDiTWrapper, reach into .model ──
                inner = getattr(comp.module, 'model', comp.module)
                for attr in ('enable_gradient_checkpointing', 'gradient_checkpointing_enable'):
                    if hasattr(inner, attr):
                        getattr(inner, attr)()
                        break
                else:
                    for m in inner.modules():
                        if hasattr(m, 'gradient_checkpointing'):
                            m.gradient_checkpointing = True

    def _trainable_params(self):
        if self.lora_injectors:
            params = []
            for inj in self.lora_injectors.values():
                params.extend(inj.get_trainable_parameters())
            return params
        return [p for p in self.model.parameters() if p.requires_grad]

    def _state(self) -> Dict:
        s = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'mode': self.mode,
            'run_dir': self.run_dir,
        }
        # Collect info from all LoRA injectors
        if self.lora_injectors:
            lora_info = {}
            for name, inj in self.lora_injectors.items():
                if hasattr(inj, 'get_info'):
                    lora_info[name] = inj.get_info()
            if lora_info:
                s['lora_info'] = lora_info
        return s

    def _fire(self, method, *a, **kw):
        for cb in self.callbacks:
            if hasattr(cb, method):
                getattr(cb, method)(*a, **kw)

    # ── Forward ──
    def _forward(self, batch):
        use_amp = self.amp_dtype != torch.float32 and self.device.type == 'cuda'
        ctx = autocast(dtype=self.amp_dtype) if use_amp else _NullCtx()
        with ctx:
            if hasattr(self.model, 'forward_pass') and callable(self.model.forward_pass):
                out = self.model.forward_pass(batch)
                if isinstance(out, dict) and 'loss' in out:
                    return out['loss'], out
                if isinstance(out, dict) and 'predictions' in out:
                    preds = out['predictions']
                    tgt = batch.get('target', batch.get('labels'))
                    if self.loss_fn is not None and tgt is not None:
                        return self.loss_fn.compute(preds, tgt), {'predictions': preds}
                    return preds, out
                if out is not None:
                    return (out, {})
                # fall through if None
    
            if self.loss_fn is not None:
                inp = batch.get('input', batch.get('pixel_values'))
                tgt = batch.get('target', batch.get('labels'))
                if inp is not None and tgt is not None:
                    preds = self.model(inp)
                    return self.loss_fn.compute(preds, tgt), {'predictions': preds}
    
            if 'input' in batch:
                out = self.model(batch['input'], batch.get('target'))
                return (out[0], {}) if isinstance(out, tuple) else (out, {})
    
            raise ValueError(  # ← now INSIDE the with block
                f"Cannot forward from batch keys {list(batch.keys())}. "
                f"loss_fn={'set' if self.loss_fn else 'None'}, "
                f"has forward_pass={hasattr(self.model, 'forward_pass')}"
            )

    # ── Train loop ──
    def train(self, **kw) -> Dict:
        if self.low_vram and self.component_bundle:
            for comp in self.component_bundle:
                if comp.module is not None:
                    # ── CHANGE: ALL components start on CPU in low_vram mode ──
                    # frozen components are moved to GPU only when needed (encode step)
                    # and offloaded before the trainable DiT forward
                    comp.module.to(self._offload_device)
        else:
            self.model.to(self.device)

        for inj in self.lora_injectors.values():
            if hasattr(inj, 'lora_layers'):
                for lora_layer in inj.lora_layers.values():
                    # In low_vram mode LoRA stays on CPU with the base model
                    # _component_on_device will move everything together
                    target = self.device if not self.low_vram else self._offload_device
                    lora_layer.to(target)

        self._fire('on_train_start', self._state())
        log = {'losses': [], 'val_losses': [], 'learning_rates': [], 'metrics': []}

        for epoch in range(self.epochs):
            if self._check_stop():  # ← ADD THIS
                break
            self.current_epoch = epoch
            self._fire('on_epoch_start', epoch, self._state())
            self._update_component_freeze_state(epoch)

            ep_loss, ep_metrics = self._train_epoch(log)
            log['losses'].append({'epoch': epoch, 'loss': ep_loss, 'step': self.global_step})

            state = self._state()
            state['epoch_loss'] = ep_loss
            if ep_metrics.get('train_accuracy') is not None:
                state['train_accuracy'] = ep_metrics['train_accuracy']

            # Validation
            if self.val_dataloader is not None:
                vr = self.validate()
                vl = vr.get('loss', 0)
                va = vr.get('accuracy')
                log['val_losses'].append({'epoch': epoch, 'loss': vl, 'step': self.global_step})
                log['metrics'].append({'epoch': epoch, **vr})
                state['val_loss'] = vl
                state['val_metrics'] = vr

                improved = False
                if vl < self.best_val_loss:
                    self.best_val_loss = vl
                    improved = True
                if va is not None and va > self.best_val_accuracy:
                    self.best_val_accuracy = va
                    improved = True
                if improved:
                    self._save(os.path.join(self.run_dir, 'best_model.pt'), tag='best')

            self._fire('on_epoch_end', epoch, state)

            if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                self._save(
                    os.path.join(self.checkpoints_dir, f'epoch_{epoch+1}.pt'),
                    tag=f'epoch_{epoch+1}',
                )

        self._fire('on_train_end', self._state())
        self._save(os.path.join(self.run_dir, 'final_model.pt'), tag='final')

        summary = {
            'mode': self.mode,
            'total_steps': self.global_step,
            'total_epochs': self.epochs,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'run_dir': self.run_dir,
        }
        if self.component_bundle:
            summary['components'] = self.component_bundle.info()
        with open(os.path.join(self.run_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return log

    def _train_epoch(self, log):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n = 0
        running_loss = 0.0
        running_n = 0

        for bi, batch in enumerate(self.train_dataloader):
            self._fire('on_step_start', self.global_step, self._state())
            batch = self._to_device(batch)

            loss, extras = self._forward(batch)
            scaled_loss = loss / self.gradient_accumulation_steps
            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (bi + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_gradients()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self._clip_gradients()
                    self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            sl = loss.item()
            total_loss += sl
            n += 1
            running_loss += sl
            running_n += 1

            if 'predictions' in extras:
                tgt = batch.get('target', batch.get('labels'))
                if tgt is not None and extras['predictions'].dim() >= 2:
                    total_correct += (extras['predictions'].argmax(-1) == tgt).sum().item()
                    total_samples += tgt.numel()

            state = self._state()
            state['lr'] = self.optimizer.param_groups[0]['lr']
            smoothed = running_loss / running_n if running_n else sl
            state['smoothed_loss'] = smoothed
            if total_samples > 0:
                state['running_accuracy'] = total_correct / total_samples
            self._fire('on_step_end', self.global_step, sl, state)
            log['learning_rates'].append({
                'step': self.global_step,
                'lr': self.optimizer.param_groups[0]['lr'],
            })

            if running_n >= 50:
                running_loss = 0.0
                running_n = 0

            # ── Manual controls check ──
            if self._check_stop():
                if self.config.get('save_on_interrupt', True):
                    self._save(
                        os.path.join(self.checkpoints_dir, f'interrupted_step_{self.global_step}.pt'),
                        tag='interrupted',
                    )
                break

            if self._save_requested:
                save_path = os.path.join(self.checkpoints_dir, f'manual_step_{self.global_step}.pt')
                self._save(save_path, tag=f'manual_step_{self.global_step}')
                self._rotate_checkpoints(save_path)
                self._save_requested = False

            # Sample check (automatic + manual)
            self._do_sample(self.global_step, self.current_epoch)

            if (self.save_every_n_steps > 0
                    and self.global_step > 0
                    and self.global_step % self.save_every_n_steps == 0):
                self._save(
                    os.path.join(self.checkpoints_dir, f'step_{self.global_step}.pt'),
                    tag=f'step_{self.global_step}',
                )

            if (self.eval_every_n_steps > 0
                    and self.global_step > 0
                    and self.global_step % self.eval_every_n_steps == 0
                    and self.val_dataloader):
                vr = self.validate()
                log['val_losses'].append({
                    'step': self.global_step,
                    'epoch': self.current_epoch,
                    'loss': vr.get('loss', 0),
                })

        avg = total_loss / max(n, 1)
        m = {'train_loss': avg}
        if total_samples > 0:
            m['train_accuracy'] = total_correct / total_samples
        return avg, m

    def validate(self, **kw):
        self.model.eval()
        tl = 0.0
        tc = 0
        ts = 0
        n = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._to_device(batch)
                loss, extras = self._forward(batch)
                tl += loss.item()
                n += 1
                if 'predictions' in extras:
                    tgt = batch.get('target', batch.get('labels'))
                    if tgt is not None and extras['predictions'].dim() >= 2:
                        tc += (extras['predictions'].argmax(-1) == tgt).sum().item()
                        ts += tgt.numel()
        self.model.train()
        r = {'loss': tl / max(n, 1)}
        if ts > 0:
            r['accuracy'] = tc / ts
        return r

    # ── Save / Load ──

    def _save(self, path: str, tag: str = ''):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # Save LoRA weights per component
        if self.lora_injectors:
            for comp_name, inj in self.lora_injectors.items():
                suffix = f'_{comp_name}' if len(self.lora_injectors) > 1 else ''
                lora_path = path.replace('.pt', f'{suffix}.safetensors')
                inj.save_weights(lora_path)

        # Save non-LoRA trainable weights
        has_non_lora = False
        if self.component_bundle:
            for comp in self.component_bundle.get_trainable():
                if comp.name not in self.lora_injectors and comp.module is not None:
                    has_non_lora = True
                    break
        
        if has_non_lora or not self.lora_injectors:
            state_dict = {}
            if self.component_bundle:
                for comp in self.component_bundle.get_trainable():
                    if comp.name not in self.lora_injectors and comp.module is not None:
                        prefix = f'{comp.name}.' if len(self.component_bundle) > 1 else ''
                        for k, v in comp.module.state_dict().items():
                            state_dict[f'{prefix}{k}'] = v
            else:
                state_dict = self.model.state_dict()
        
            # ── CHANGE: save model weights as safetensors ──
            from safetensors.torch import save_file
            sf_path = path.replace('.pt', '.safetensors')
            save_file({k: v.contiguous().cpu() for k, v in state_dict.items()}, sf_path)
        
            # Optimizer + training state still needs torch.save (no safetensors equivalent)
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_accuracy,
                'mode': self.mode,
            }, path.replace('.pt', '_trainer.pt'))

        # Metadata
        meta = {
            'tag': tag,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'lora_components': list(self.lora_injectors.keys()),
        }
        if self.component_bundle:
            meta['components'] = [c.name for c in self.component_bundle]
        with open(path.rsplit('.', 1)[0] + '_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        self._fire('on_checkpoint_saved', path, tag, self.global_step, self.current_epoch)
        if tag.startswith('step_') or tag.startswith('epoch_'):
            self._rotate_checkpoints(path)

    def load_checkpoint(self, path: str):
        # Load per-component LoRA
        if self.lora_injectors:
            for comp_name, inj in self.lora_injectors.items():
                suffix = f'_{comp_name}' if len(self.lora_injectors) > 1 else ''
                lora_path = path.replace('.pt', f'{suffix}.safetensors')
                if os.path.exists(lora_path):
                    inj.load_weights(lora_path)

        # Load non-LoRA state
        if os.path.exists(path):
            # ── CHANGE: load model weights from safetensors ──
            from safetensors.torch import load_file
            sf_path = path.replace('.pt', '.safetensors')
            trainer_path = path.replace('.pt', '_trainer.pt')
        
            if os.path.exists(sf_path):
                model_sd = load_file(sf_path, device='cpu')
                if self.component_bundle and len(self.component_bundle) > 1:
                    for comp in self.component_bundle.get_trainable():
                        if comp.name in self.lora_injectors:
                            continue
                        prefix = f'{comp.name}.'
                        comp_sd = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
                        if comp_sd and comp.module:
                            comp.module.load_state_dict(comp_sd, strict=False)
                else:
                    self.model.load_state_dict(model_sd, strict=False)
        
            if os.path.exists(trainer_path):
                ckpt = torch.load(trainer_path, map_location='cpu', weights_only=False)
                if 'optimizer_state_dict' in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    except Exception:
                        pass
                self.global_step = ckpt.get('global_step', 0)
                self.current_epoch = ckpt.get('epoch', 0)
                self.best_val_loss = ckpt.get('best_val_loss', float('inf'))

    def cleanup(self):
        """Release GPU memory. Call on any exit path."""
        try:
            if self.component_bundle:
                for comp in self.component_bundle:
                    if comp.module is not None:
                        comp.module.to('cpu')
            elif self.model is not None:
                self.model.to('cpu')
            
            # Move LoRA layers off GPU too
            for inj in self.lora_injectors.values():
                if hasattr(inj, 'lora_layers'):
                    for lora in inj.lora_layers.values():
                        lora.to('cpu')
            
            if self.scaler:
                del self.scaler
                self.scaler = None

            if hasattr(self, '_layer_streamer') and self._layer_streamer is not None:
                self._layer_streamer.disable()
                
        except Exception as e:
            print(f"[Cleanup] Warning: {e}")
        finally:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("[Cleanup] VRAM released.")

    # ── Device / Freeze / Grad ──
    @contextmanager
    def _component_on_device(self, comp):
        try:
            comp.module.to(self.device)
            if hasattr(comp.module, '_fix_all_tensors'):
                comp.module._fix_all_tensors(comp.module)
            
            # ── NEW: also move this component's LoRA layers to GPU ──
            comp_name = comp.name
            if comp_name in self.lora_injectors:
                inj = self.lora_injectors[comp_name]
                for lora_layer in inj.lora_layers.values():
                    lora_layer.to(self.device)
            
            yield
        finally:
            comp.module.to(self._offload_device)
            
            # ── NEW: offload LoRA back to CPU too ──
            if comp.name in self.lora_injectors:
                inj = self.lora_injectors[comp.name]
                for lora_layer in inj.lora_layers.values():
                    lora_layer.to(self._offload_device)
            
            torch.cuda.empty_cache()  # ← also add this

    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return {'input': batch[0].to(self.device), 'target': batch[1].to(self.device)}
            return {'input': batch[0].to(self.device)}
        if isinstance(batch, dict):
            return {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, torch.Tensor):
            return {'input': batch.to(self.device)}
        return batch

    def _update_component_freeze_state(self, epoch: int):
        schedule = getattr(self, '_freeze_schedule', {})
        if not schedule or not self.component_bundle:
            return

        for comp_name, sched in schedule.items():
            if epoch == sched['freeze_epochs']:
                comp = self.component_bundle.get(comp_name)
                if comp is None:
                    continue

                strategy = sched.get('strategy', 'full')
                patterns = sched.get('unfreeze_patterns')

                if strategy == 'finetune' and patterns:
                    comp.unfreeze(patterns=patterns)
                else:
                    comp.unfreeze()

                new_params = [p for p in comp.module.parameters() if p.requires_grad]
                if new_params:
                    comp_lr = (comp.config.get('training', {}) or {}).get('lr') \
                              or self.optimizer.param_groups[0]['lr']
                    self.optimizer.add_param_group({
                        'params': new_params,
                        'lr': comp_lr,
                    })

    def _clip_gradients(self):
        """Clip gradients per param group. unscale_ already called by caller if using scaler."""
        global_norm = self.max_grad_norm
        for group in self.optimizer.param_groups:
            comp_norm = group.get('_max_grad_norm') or global_norm
            if comp_norm and group['params']:
                torch.nn.utils.clip_grad_norm_(group['params'], comp_norm)

    def request_save(self):
        """API calls this → trainer saves at next step boundary."""
        self._save_requested = True

    def request_sample(self, request_data=None):
        """API calls this → trainer generates samples at next step boundary."""
        self._sample_requested = True
        self._sample_request_data = request_data

    def _check_stop(self) -> bool:
        """Check if stop was requested."""
        if self.stop_event is not None and self.stop_event.is_set():
            return True
        return False

    def _do_sample(self, step: int, epoch: int, force: bool = False):
        """Generate samples if conditions are met."""
        if self.sampler is None:
            return

        should = force or self._sample_requested or self.sampler.should_sample(step, epoch)
        if not should:
            return

        # Switch to eval
        self.model.eval()

        # Remove training adapter for clean sampling
        if self.training_adapter is not None:
            self.training_adapter.disable()

        request = None
        if self._sample_requested and self._sample_request_data:
            from backend.pipelines.base_pipeline import SampleRequest
            rd = self._sample_request_data
            request = SampleRequest(
                prompts=rd.get('prompts', self.sampler.prompts),
                negative_prompts=rd.get('negative_prompts'),
                width=rd.get('width', self.sampler.width),
                height=rd.get('height', self.sampler.height),
                num_steps=rd.get('num_steps', self.sampler.num_steps),
                guidance_scale=rd.get('guidance_scale', self.sampler.guidance_scale),
                seed=rd.get('seed', self.sampler.seed),
                sampler=rd.get('sampler', self.sampler.sampler_name),
            )

        result = self.sampler.generate(step, epoch, request=request)

        # Log sample images to TensorBoard
        self._log_samples_to_tb(result, step)

        if self.training_adapter is not None:
            self.training_adapter.enable()

        # Reset manual request flag
        self._sample_requested = False
        self._sample_request_data = None

        # Back to train
        self.model.train()

    def _log_samples_to_tb(self, result, step):
        """Log sample images to TensorBoard if available."""
        for cb in self.callbacks:
            if hasattr(cb, 'writer') and cb.writer is not None:
                try:
                    import torchvision.transforms.functional as TF
                    for i, img in enumerate(result.images):
                        if hasattr(img, 'convert'):  # PIL Image
                            tensor = TF.to_tensor(img)
                            cb.writer.add_image(
                                f'samples/{result.prompts[i][:50]}',
                                tensor, step
                            )
                    cb.writer.flush()
                except Exception as e:
                    print(f"[Sampler] TensorBoard image logging failed: {e}")
                break

    def _rotate_checkpoints(self, new_path: str):
        """Keep only the last N checkpoints."""
        if self.keep_last_n <= 0:
            return

        self._checkpoint_history.append(new_path)

        while len(self._checkpoint_history) > self.keep_last_n:
            old_path = self._checkpoint_history.pop(0)
            # Delete checkpoint file and associated files
            for ext_path in [old_path,
                             old_path.replace('.pt', '.safetensors'),
                             old_path.rsplit('.', 1)[0] + '_meta.json']:
                if os.path.exists(ext_path):
                    try:
                        os.remove(ext_path)
                        print(f"[Checkpoint] Rotated out: {ext_path}")
                    except Exception:
                        pass
            # Also try component-specific LoRA files
            if self.lora_injectors:
                for comp_name in self.lora_injectors:
                    suffix = f'_{comp_name}' if len(self.lora_injectors) > 1 else ''
                    lora_path = old_path.replace('.pt', f'{suffix}.safetensors')
                    if os.path.exists(lora_path):
                        try:
                            os.remove(lora_path)
                        except Exception:
                            pass


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
