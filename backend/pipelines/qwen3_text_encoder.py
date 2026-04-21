"""
Qwen3-4B text encoder wrapper for ZImageTurbo.

Weights come from local .safetensors file.
Tokenizer + config are downloaded from HuggingFace on first run,
then cached locally next to the model file so subsequent runs are offline.
"""
import os
import torch
import torch.nn as nn


HF_MODEL_ID = "Qwen/Qwen3-4B"


class Qwen3TextEncoder(nn.Module):

    def __init__(self, path: str, dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None, max_length: int = 512):
        super().__init__()
        self._dtype = dtype
        self._device = device or torch.device('cpu')
        self.max_length = max_length
        self._model, self._tokenizer = self._load(path)

    def _load(self, path: str):
        try:
            import tiktoken  # noqa: F401 — required by Qwen tokenizer
        except ImportError:
            raise RuntimeError(
                "Qwen3 tokenizer requires 'tiktoken'. "
                "Install it with: pip install tiktoken"
            )

        from transformers import AutoTokenizer, AutoConfig, AutoModel
        from safetensors.torch import load_file

        # Resolve directories
        if os.path.isfile(path):
            model_dir = os.path.dirname(path)
            weights_path = path
        else:
            # path is a directory — look for the safetensors inside it
            model_dir = path
            weights_path = self._find_weights(model_dir)

        tokenizer_cache = os.path.join(model_dir, '_tokenizer_cache')
        config_cache    = os.path.join(model_dir, '_config_cache')

        # ── Tokenizer ────────────────────────────────────────────────────────
        tokenizer_ready = os.path.exists(
            os.path.join(tokenizer_cache, 'tokenizer.json')
        )
        if tokenizer_ready:
            print(f"[Qwen3] Loading cached tokenizer from '{tokenizer_cache}'")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_cache, trust_remote_code=True
            )
        else:
            print(f"[Qwen3] Downloading tokenizer from HuggingFace '{HF_MODEL_ID}'...")
            os.makedirs(tokenizer_cache, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(
                HF_MODEL_ID,
                trust_remote_code=True,
            )
            tokenizer.save_pretrained(tokenizer_cache)
            print(f"[Qwen3] Tokenizer cached to '{tokenizer_cache}'")

        # ── Config ───────────────────────────────────────────────────────────
        config_ready = os.path.exists(
            os.path.join(config_cache, 'config.json')
        )
        if config_ready:
            print(f"[Qwen3] Loading cached config from '{config_cache}'")
            config = AutoConfig.from_pretrained(
                config_cache, trust_remote_code=True
            )
        else:
            print(f"[Qwen3] Downloading config from HuggingFace '{HF_MODEL_ID}'...")
            os.makedirs(config_cache, exist_ok=True)
            config = AutoConfig.from_pretrained(
                HF_MODEL_ID,
                trust_remote_code=True,
            )
            config.save_pretrained(config_cache)
            print(f"[Qwen3] Config cached to '{config_cache}'")

        # ── Model ─────────────────────────────────────────────────────────────
        print(f"[Qwen3] Building model from config...")
        model = AutoModel.from_config(config, trust_remote_code=True)

        model.gradient_checkpointing_disable()
        
        # Also patch any layers that still have it enabled
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False

        print(f"[Qwen3] Loading weights from '{weights_path}'")
        sd = load_file(weights_path)

        # Qwen3 weights may have a 'model.' prefix — strip it if present
        sample_key = next(iter(sd))
        if sample_key.startswith('model.'):
            print("[Qwen3] Stripping 'model.' prefix from state dict keys")
            sd = {k[len('model.'):]: v for k, v in sd.items()}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Qwen3] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[Qwen3] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        model = model.to(self._device, self._dtype).eval()

        # Always frozen — text encoder is never trained
        for p in model.parameters():
            p.requires_grad = False

        print(f"[Qwen3] Ready — {sum(p.numel() for p in model.parameters()):,} params")
        return model, tokenizer

    def _find_weights(self, model_dir: str) -> str:
        """Find the Qwen3 safetensors file in a directory."""
        candidates = [
            'qwen_3_4b.safetensors',
            'qwen3_4b.safetensors',
            'model.safetensors',
        ]
        for name in candidates:
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                return p

        # Last resort: any .safetensors that isn't ae/vae/dit
        for fname in os.listdir(model_dir):
            if fname.endswith('.safetensors') and 'qwen' in fname.lower():
                return os.path.join(model_dir, fname)

        raise FileNotFoundError(
            f"Could not find Qwen3 weights in '{model_dir}'. "
            f"Files present: {os.listdir(model_dir)}"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Returns [seq_len, hidden_dim] — variable length, Z-Image style.
        Uses second-to-last hidden state with chat template.
        """
        if not text or not text.strip():
            text = "."
    
        messages = [{"role": "user", "content": text}]
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self._tokenizer(
            [formatted],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
    
        mask = inputs.attention_mask.bool()
        out = self._model(
            input_ids=inputs.input_ids,
            attention_mask=mask,
            output_hidden_states=True,
        )
        # Z-Image uses second-to-last hidden state, masked to actual tokens
        return out.hidden_states[-2][0][mask[0]]  # [seq_len, hidden_dim]

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
