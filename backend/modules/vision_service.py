"""
Abstracted vision/language model service.

Supports pluggable model backends. Currently: Qwen3-VL via transformers.
Designed so you can add new backends (e.g. LLaVA, CogVLM, API-based) by
subclassing VisionModelBackend.
"""

import os
import re
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class CaptionRequest:
    image_path: str
    prompt: str = "Describe this image in detail."
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class ConceptExtractionRequest:
    captions: List[str]
    categories: List[str] = field(default_factory=lambda: [
        "attributes", "actions", "settings", "style", "composition",
    ])
    prompt_template: Optional[str] = None


# ── Abstract backend ──

class VisionModelBackend(ABC):
    """Base class for vision-language model backends."""

    @abstractmethod
    def load(self, model_id: str, **kwargs) -> None: ...

    @abstractmethod
    def caption_image(self, request: CaptionRequest) -> str: ...

    @abstractmethod
    def text_completion(self, prompt: str, config: GenerationConfig) -> str: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def get_info(self) -> Dict[str, Any]: ...


# ── Qwen3-VL backend ──

class Qwen3VLBackend(VisionModelBackend):
    """Qwen3-VL via HuggingFace transformers."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_id = None
        self.device = None

    def load(self, model_id: str, **kwargs):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        dtype = kwargs.get("dtype", torch.bfloat16)
        device_map = kwargs.get("device_map", "auto")
        attn_impl = kwargs.get("attn_implementation", None)

        load_kwargs = dict(
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        # Tune thread count
        cpu_count = os.cpu_count() or 4
        half = max(1, cpu_count // 2)
        os.environ["MKL_NUM_THREADS"] = str(half)
        os.environ["OMP_NUM_THREADS"] = str(half)
        torch.set_num_threads(half)

        print(f"[VisionService] Loading {model_id}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self.device = next(self.model.parameters()).device
        print(f"[VisionService] Loaded on {self.device}")

    def caption_image(self, request: CaptionRequest) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": request.image_path},
                {"type": "text", "text": request.prompt},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=request.generation_config.max_new_tokens,
                temperature=request.generation_config.temperature,
                top_p=request.generation_config.top_p,
                do_sample=request.generation_config.do_sample,
            )

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text

    def text_completion(self, prompt: str, config: GenerationConfig) -> str:
        """Text-only completion (for concept extraction etc.)."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self):
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.model_id = None
            torch.cuda.empty_cache()
            print("[VisionService] Model unloaded")

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": "qwen3vl",
            "model_id": self.model_id,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
        }

class Qwen25VLBackend(VisionModelBackend):
    """Qwen2.5-VL via HuggingFace transformers."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_id = None
        self.device = None

    def load(self, model_id: str, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        dtype = kwargs.get("dtype", "auto")
        device_map = kwargs.get("device_map", "auto")

        load_kwargs = dict(
            torch_dtype=dtype,
            device_map=device_map,
        )

        cpu_count = os.cpu_count() or 4
        half = max(1, cpu_count // 2)
        os.environ["MKL_NUM_THREADS"] = str(half)
        os.environ["OMP_NUM_THREADS"] = str(half)
        torch.set_num_threads(half)

        print(f"[VisionService] Loading {model_id}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self.device = next(self.model.parameters()).device
        self._process_vision_info = process_vision_info
        print(f"[VisionService] Loaded on {self.device}")

    def _run_inference(self, messages: list, config: GenerationConfig) -> str:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def caption_image(self, request: CaptionRequest) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": request.image_path},
                {"type": "text", "text": request.prompt},
            ],
        }]
        return self._run_inference(messages, request.generation_config)

    def text_completion(self, prompt: str, config: GenerationConfig) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._run_inference(messages, config)

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self):
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.model_id = None
            torch.cuda.empty_cache()
            print("[VisionService] Model unloaded")

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": "qwen25vl",
            "model_id": self.model_id,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
        }


# ── Registry ──

BACKENDS = {
    "qwen3vl": Qwen3VLBackend,
    "qwen25vl": Qwen25VLBackend,
    # Future: "llava": LLaVABackend, "cogvlm": CogVLMBackend, etc.
}

# Global singleton
_active_backend: Optional[VisionModelBackend] = None


def get_backend() -> Optional[VisionModelBackend]:
    return _active_backend


def load_backend(backend_name: str, model_id: str, **kwargs) -> VisionModelBackend:
    global _active_backend
    if _active_backend and _active_backend.is_loaded():
        _active_backend.unload()

    cls = BACKENDS.get(backend_name)
    if not cls:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(BACKENDS.keys())}")

    _active_backend = cls()
    _active_backend.load(model_id, **kwargs)
    return _active_backend


def unload_backend():
    global _active_backend
    if _active_backend:
        _active_backend.unload()
        _active_backend = None
