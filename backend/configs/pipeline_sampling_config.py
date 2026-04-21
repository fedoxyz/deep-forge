"""
Config additions for pipeline, sampling, and checkpoint management.

Add these to DEFAULT_CONFIG in config_manager.py
"""

# ══════════════════════════════════════════════════════════════
# ADD to DEFAULT_CONFIG dict in config_manager.py:
# ══════════════════════════════════════════════════════════════

PIPELINE_AND_SAMPLING_CONFIG = {
    # ── Pipeline (NEW) ──
    # Defines which diffusion pipeline to use for training + inference.
    # Required for diffusion model training. Ignored for non-diffusion tasks.
    'pipeline': {
        'name': None,              # 'zimage_turbo' | 'flux' | 'sdxl' | None (auto-detect)
        'params': {},              # Pipeline-specific params (e.g. shift, loss_type)
    },

    # ── Sampling during training (NEW) ──
    'sampling': {
        'enabled': False,
        'every_n_steps': 500,      # Generate samples every N training steps
        'every_n_epochs': 0,       # Or every N epochs (0 = disabled)
        'prompts': [],             # List of prompts to generate
        'negative_prompts': [],    # Corresponding negative prompts
        'width': 1024,
        'height': 1024,
        'num_steps': 8,            # Inference steps for sampling
        'guidance_scale': 0.0,     # CFG scale (0.0 for distilled models like Turbo)
        'sampler': 'euler',        # 'euler' | 'euler_a' | 'dpm++'
        'seed': 42,                # Fixed seed for reproducible samples
    },

    # ── Checkpoint management (additions to existing 'training' section) ──
    # Add these keys to the existing 'training' dict:
    'training_checkpoint_additions': {
        'keep_last_n_checkpoints': 5,   # Auto-delete older checkpoints (0 = keep all)
        'save_on_interrupt': True,       # Save checkpoint when training is stopped
    },
}


# ══════════════════════════════════════════════════════════════
# Example Z-Image Turbo character LoRA config:
# ══════════════════════════════════════════════════════════════

EXAMPLE_ZIMAGE_TURBO_CHARACTER_CONFIG = {
    'general': {
        'mode': 'lora',
    },

    'pipeline': {
        'name': 'zimage_turbo',
        'params': {
            'shift': 1.0,
            'loss_type': 'mse',      # or 'huber'
            'snr_gamma': None,        # set to 5.0 for min-SNR weighting
        },
        'training_adapter': {
            'path': 'adapters/zimage_turbo_training_adapter_v2.safetensors',
        },
    },

    'model': {
        'base_dir': '/models',
        'components': [
            {
                'name': 'dit',
                'source': 'file',
                'path': 'diffusion_models/z_image_turbo_bf16.safetensors',
                'role': 'denoiser',
                'dtype': 'bfloat16',
                'execution_order': 2,
                'training': {
                    'strategy': 'lora',
                    'lr': 5e-5,
                    'lora': {
                        'rank': 16,
                        'alpha': 16,
                        'dropout': 0.0,
                        'target_patterns': [
                            # Middle attention blocks only — preserves distillation
                            r'transformer_blocks\.([89]|1[0-9]|2[0-2])\.attention\.(to_q|to_k|to_v|to_out)',
                        ],
                    },
                },
                'forward': {
                    'input_key': 'noisy_latents',
                    'output_key': 'model_pred',
                    'cache_output': False,
                },
            },
            {
                'name': 'text_encoder',
                'source': 'file',
                'path': 'text_encoders/qwen_3_4b.safetensors',
                'role': 'text_encoder',
                'dtype': 'bfloat16',
                'execution_order': 0,
                'training': {
                    'strategy': 'frozen',
                },
                'forward': {
                    'input_key': 'caption',
                    'output_key': 'prompt_embeds',
                    'no_grad': True,
                    'cache_output': True,
                },
            },
            {
                'name': 'vae',
                'source': 'file',
                'path': 'vae/ae.safetensors',
                'role': 'vae',
                'dtype': 'bfloat16',
                'execution_order': 1,
                'training': {
                    'strategy': 'frozen',
                },
                'forward': {
                    'input_key': 'pixel_values',
                    'output_key': 'latents',
                    'no_grad': True,
                    'cache_output': False,
                },
            },
        ],
    },

    'lora': {
        'rank': 16,
        'alpha': 16,
        'dropout': 0.0,
        'init_reversed': True,
    },

    'dataset': {
        'path': './datasets/my_character',
        'batch_size': 1,
        'num_workers': 2,
        'center_crop': True,
        'random_flip': 0.5,
    },

    'training': {
        'epochs': 50,
        'gradient_accumulation_steps': 4,
        'max_grad_norm': 1.0,
        'mixed_precision': 'bf16',
        'gradient_checkpointing': True,
        'save_every_n_steps': 500,
        'eval_every_n_steps': 0,
        'keep_last_n_checkpoints': 5,
        'save_on_interrupt': True,
        'seed': 42,
    },

    'optimizer': {
        'name': 'adamw',
        'lr': 5e-5,
        'weight_decay': 0.01,
        'betas': [0.9, 0.999],
    },

    'scheduler': {
        'name': 'cosine_warmup',
        'warmup_steps': 100,
    },

    'loss': {
        'name': 'mse',
    },

    'sampling': {
        'enabled': True,
        'every_n_steps': 500,
        'prompts': [
            'portrait of <character> smiling, professional photo, studio lighting',
            '<character> walking in a park, sunny day, full body shot',
            'close-up of <character>, dramatic lighting, cinematic',
        ],
        'width': 1024,
        'height': 1024,
        'num_steps': 8,
        'guidance_scale': 0.0,
        'sampler': 'euler',
        'seed': 42,
    },

    'logging': {
        'tensorboard': True,
        'print_every': 10,
    },

    'output': {
        'dir': './outputs',
        'save_format': 'safetensors',
    },
}
