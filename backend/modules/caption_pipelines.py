"""
High-level pipelines that use VisionModelBackend for dataset tasks.

- caption_single / caption_batch: Generate captions for images
- extract_concepts_llm: Use LLM to extract structured concepts from captions
"""

import json
import re
from typing import Any, Dict, List, Optional
from backend.modules.vision_service import (
    get_backend, CaptionRequest, GenerationConfig, ConceptExtractionRequest,
)


DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail for AI training. Include: the subject's appearance "
    "(hair, clothing, body), their pose and expression, the setting/environment, "
    "lighting conditions, camera angle, and any notable details. "
    "Be specific and use natural language. One paragraph."
)

DEFAULT_CONCEPT_PROMPT_TEMPLATE = """Analyze these image captions from a training dataset. Extract all recurring concepts and categorize them.

Categories: {categories}

Captions:
{captions_block}

For each concept found, output a JSON array of objects with these fields:
- "phrase": the concept phrase
- "category": one of the categories above
- "count": how many captions contain this concept
- "caption_indices": list of 0-based indices of captions containing it

Important:
- Merge similar phrasings (e.g. "soft light" and "softly lit" = same concept)
- Ignore trigger/token words that appear in every caption (like subject identifiers)
- Focus on meaningful visual concepts, not filler words

Output ONLY the JSON array, no other text."""


def caption_single(
    image_path: str,
    prompt: str = DEFAULT_CAPTION_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Caption a single image."""
    backend = get_backend()
    if not backend or not backend.is_loaded():
        raise RuntimeError("No vision model loaded. Load one first via /api/vision/load")

    return backend.caption_image(CaptionRequest(
        image_path=image_path,
        prompt=prompt,
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ),
    ))


def caption_batch(
    image_paths: List[str],
    prompt: str = DEFAULT_CAPTION_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    on_progress: Optional[callable] = None,
) -> List[Dict[str, str]]:
    """Caption multiple images sequentially. Returns list of {path, caption}."""
    results = []
    for i, path in enumerate(image_paths):
        try:
            cap = caption_single(path, prompt, max_new_tokens, temperature)
            results.append({"path": path, "caption": cap, "error": None})
        except Exception as e:
            results.append({"path": path, "caption": "", "error": str(e)})
        if on_progress:
            on_progress(i + 1, len(image_paths))
    return results


def extract_concepts_llm(
    request: ConceptExtractionRequest,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Use the loaded LLM to extract structured concepts from captions.
    Chunks captions if needed to stay within context limits.
    """
    backend = get_backend()
    if not backend or not backend.is_loaded():
        raise RuntimeError("No vision model loaded")

    categories_str = ", ".join(request.categories)
    template = request.prompt_template or DEFAULT_CONCEPT_PROMPT_TEMPLATE

    # Chunk captions to ~30 at a time to avoid context overflow
    CHUNK_SIZE = 30
    all_concepts: Dict[str, Dict] = {}

    for chunk_start in range(0, len(request.captions), CHUNK_SIZE):
        chunk = request.captions[chunk_start:chunk_start + CHUNK_SIZE]
        captions_block = "\n".join(
            f"[{chunk_start + i}] {c}" for i, c in enumerate(chunk)
        )

        prompt = template.format(
            categories=categories_str,
            captions_block=captions_block,
        )

        raw = backend.text_completion(prompt, GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        ))

        # Parse JSON from response
        parsed = _extract_json_array(raw)
        for concept in parsed:
            phrase = concept.get("phrase", "").strip().lower()
            if not phrase:
                continue
            if phrase in all_concepts:
                existing = all_concepts[phrase]
                existing["count"] = existing.get("count", 0) + concept.get("count", 0)
                existing["caption_indices"].extend(concept.get("caption_indices", []))
            else:
                all_concepts[phrase] = {
                    "phrase": phrase,
                    "category": concept.get("category", "unknown"),
                    "count": concept.get("count", 0),
                    "caption_indices": concept.get("caption_indices", []),
                }

    # Sort by count
    result = sorted(all_concepts.values(), key=lambda c: c["count"], reverse=True)
    return result


def _extract_json_array(text: str) -> List[Dict]:
    """Try to extract a JSON array from LLM output (tolerant of markdown fences)."""
    # Strip markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Find the array
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        return []

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
