"""
Dataset manager for image+caption datasets.

Handles:
- Scanning directories for image/caption pairs
- Reading and writing captions
- Concept extraction and frequency analysis
- Co-occurrence analysis
- Thumbnail generation
"""

import os
import re
import hashlib
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
import json

from PIL import Image

DATASET_BASE_DIR = os.environ.get("DATASET_BASE_DIR", "/data/dataset")
PERSIST_FILE = os.environ.get("DATASET_PERSIST_FILE", "/data/dataset/.loaded_datasets.json")

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}


# ── Data Classes ──

@dataclass
class ImageEntry:
    """Single image+caption entry."""
    filename: str
    image_path: str
    caption_path: Optional[str]
    caption: str
    width: int
    height: int
    file_size: int  # bytes
    has_caption_file: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptInfo:
    """A discovered concept with its frequency and image associations."""
    phrase: str
    count: int
    image_indices: List[int]
    category: str = "unknown"  # attribute, action, setting, style, composition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase,
            "count": self.count,
            "image_indices": self.image_indices,
            "category": self.category,
        }


@dataclass
class DatasetInfo:
    """Full dataset scan result."""
    dataset_id: str
    directory: str
    entries: List[ImageEntry] = field(default_factory=list)
    total_images: int = 0
    total_with_captions: int = 0
    total_without_captions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "directory": self.directory,
            "total_images": self.total_images,
            "total_with_captions": self.total_with_captions,
            "total_without_captions": self.total_without_captions,
            "entries": [e.to_dict() for e in self.entries],
        }


# ── Loaded datasets cache ──
_loaded_datasets: Dict[str, DatasetInfo] = {}


def _dataset_id(directory: str) -> str:
    return hashlib.md5(os.path.abspath(directory).encode()).hexdigest()[:12]


# ── Scanning ──

def scan_dataset(directory: str) -> DatasetInfo:
    """Scan a directory for image+caption pairs. Caches result."""
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    ds_id = _dataset_id(directory)
    entries = []
    root = Path(directory)

    image_files = sorted([
        f for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    for img_path in image_files:
        caption_path = img_path.with_suffix('.txt')
        has_caption = caption_path.exists()
        caption = ""
        if has_caption:
            try:
                caption = caption_path.read_text(encoding='utf-8').strip()
            except Exception:
                caption = ""

        # Get dimensions without fully loading
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        file_size = img_path.stat().st_size

        entries.append(ImageEntry(
            filename=img_path.name,
            image_path=str(img_path),
            caption_path=str(caption_path) if has_caption else None,
            caption=caption,
            width=w,
            height=h,
            file_size=file_size,
            has_caption_file=has_caption,
        ))

    info = DatasetInfo(
        dataset_id=ds_id,
        directory=directory,
        entries=entries,
        total_images=len(entries),
        total_with_captions=sum(1 for e in entries if e.has_caption_file and e.caption),
        total_without_captions=sum(1 for e in entries if not e.has_caption_file or not e.caption),
    )

    _loaded_datasets[ds_id] = info
    _save_loaded_datasets()
    return info


def get_loaded_dataset(dataset_id: str) -> Optional[DatasetInfo]:
    return _loaded_datasets.get(dataset_id)


def get_all_loaded() -> Dict[str, Dict[str, Any]]:
    return {
        ds_id: {
            "dataset_id": ds_id,
            "directory": info.directory,
            "total_images": info.total_images,
            "total_with_captions": info.total_with_captions,
        }
        for ds_id, info in _loaded_datasets.items()
    }


def unload_dataset(dataset_id: str) -> bool:
    if dataset_id in _loaded_datasets:
        del _loaded_datasets[dataset_id]
        _save_loaded_datasets()
        return True
    return False

def _save_loaded_datasets():
    try:
        data = {ds_id: info.directory for ds_id, info in _loaded_datasets.items()}
        with open(PERSIST_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[DatasetManager] Failed to save state: {e}")

def restore_loaded_datasets():
    if not os.path.exists(PERSIST_FILE):
        return
    try:
        with open(PERSIST_FILE) as f:
            data = json.load(f)
        for _, directory in data.items():
            if os.path.isdir(directory):
                scan_dataset(directory)
        print(f"[DatasetManager] Restored {len(data)} datasets")
    except Exception as e:
        print(f"[DatasetManager] Failed to restore state: {e}")


# ── Caption editing ──

def update_caption(dataset_id: str, image_index: int, new_caption: str) -> bool:
    """Update a single image's caption. Creates .txt file if it doesn't exist."""
    ds = _loaded_datasets.get(dataset_id)
    if not ds or image_index < 0 or image_index >= len(ds.entries):
        return False

    entry = ds.entries[image_index]
    img_path = Path(entry.image_path)
    caption_path = img_path.with_suffix('.txt')

    caption_path.write_text(new_caption.strip(), encoding='utf-8')

    # Update in-memory
    entry.caption = new_caption.strip()
    entry.caption_path = str(caption_path)
    entry.has_caption_file = True

    # Recount
    ds.total_with_captions = sum(1 for e in ds.entries if e.has_caption_file and e.caption)
    ds.total_without_captions = ds.total_images - ds.total_with_captions

    return True


def batch_update_captions(
    dataset_id: str, updates: Dict[int, str]
) -> Dict[str, Any]:
    """Update multiple captions at once. Returns summary."""
    success = 0
    failed = 0
    for idx, caption in updates.items():
        if update_caption(dataset_id, int(idx), caption):
            success += 1
        else:
            failed += 1
    return {"updated": success, "failed": failed}


# ── Thumbnail generation ──

def get_thumbnail_base64(
    image_path: str, max_size: int = 256
) -> Optional[str]:
    """Generate a base64 thumbnail for an image."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=80)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


# ── Concept Analysis ──

# Common stop words to filter out
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'because', 'but', 'and', 'or', 'if', 'while', 'about', 'against',
    'up', 'down', 'this', 'that', 'these', 'those', 'it', 'its',
    'she', 'her', 'he', 'his', 'they', 'them', 'their', 'we', 'our',
    'you', 'your', 'i', 'me', 'my', 'who', 'which', 'what',
    'also', 'taken', 'looking', 'photo', 'image', 'picture', 'compressed',
}

# Category hints - phrases containing these words get auto-categorized
CATEGORY_HINTS = {
    'attribute': [
        'hair', 'eyes', 'skin', 'dress', 'wearing', 'clothes', 'outfit',
        'color', 'colour', 'red', 'blue', 'green', 'black', 'white', 'blonde',
        'brunette', 'long', 'short', 'curly', 'straight', 'tall', 'thin',
        'tattoo', 'piercing', 'makeup', 'jewelry', 'hat', 'glasses',
    ],
    'action': [
        'sitting', 'standing', 'walking', 'running', 'holding', 'looking',
        'smiling', 'laughing', 'posing', 'leaning', 'lying', 'dancing',
        'pointing', 'reaching', 'touching', 'playing', 'reading', 'eating',
    ],
    'setting': [
        'indoor', 'outdoor', 'studio', 'beach', 'forest', 'city', 'urban',
        'garden', 'room', 'street', 'park', 'mountain', 'water', 'sky',
        'building', 'house', 'office', 'kitchen', 'bedroom', 'bathroom',
        'nature', 'landscape', 'sunset', 'sunrise', 'night', 'day',
    ],
    'style': [
        'lighting', 'light', 'bokeh', 'depth', 'field', 'camera', 'lens',
        'photo', 'photograph', 'cinematic', 'dramatic', 'soft', 'hard',
        'natural', 'artificial', 'flash', 'backlit', 'rim', 'ambient',
        'warm', 'cool', 'moody', 'bright', 'dark', 'high key', 'low key',
        'film', 'grain', 'analog', 'digital', 'raw', 'edited', 'retouched',
    ],
    'composition': [
        'close-up', 'closeup', 'portrait', 'full body', 'half body',
        'headshot', 'wide', 'angle', 'overhead', 'from above', 'from below',
        'side', 'front', 'behind', 'profile', 'three-quarter', 'centered',
        'rule of thirds', 'symmetrical', 'asymmetrical', 'crop', 'frame',
    ],
}


def _categorize_phrase(phrase: str) -> str:
    """Auto-categorize a phrase based on keyword hints."""
    lower = phrase.lower()
    best_cat = "unknown"
    best_score = 0
    for cat, keywords in CATEGORY_HINTS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat if best_score > 0 else "unknown"


def _extract_ngrams(
    text: str, min_n: int = 1, max_n: int = 3
) -> List[str]:
    """Extract word n-grams from text, filtering stop words for unigrams."""
    # Normalize: lowercase, remove extra punctuation but keep hyphens
    text = re.sub(r'[^\w\s\-]', ' ', text.lower())
    words = text.split()

    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            phrase = ' '.join(gram)

            # Skip if all words are stop words
            if all(w in STOP_WORDS for w in gram):
                continue
            # For unigrams, skip stop words entirely
            if n == 1 and gram[0] in STOP_WORDS:
                continue
            # Skip very short unigrams
            if n == 1 and len(gram[0]) <= 2:
                continue

            ngrams.append(phrase)

    return ngrams

def _is_likely_trigger_word(phrase: str, all_captions: List[str], threshold: float = 0.85) -> bool:
    """
    Detect if a phrase is likely a trigger word (appears in nearly all captions
    and is not a common English word). Trigger words are subject-specific tokens
    like 'lzmtpndf' or 'ohwx' that users embed for LoRA training.
    """
    if len(phrase.split()) > 1:
        return False
    # If it appears in 85%+ of captions and isn't a real word (heuristic: short or
    # has unusual character patterns), it's probably a trigger
    count = sum(1 for c in all_captions if phrase in c.lower())
    if count / max(len(all_captions), 1) >= threshold:
        # Additional check: real common words that appear everywhere are fine
        # but nonsense tokens like 'lzmtpndf' should be flagged
        common_high_freq = {
            'woman', 'man', 'person', 'photo', 'portrait', 'wearing',
            'background', 'looking', 'camera', 'standing', 'sitting',
        }
        if phrase.lower() not in common_high_freq:
            return True
    return False

def analyze_concepts(
    dataset_id: str,
    min_frequency: int = 2,
    max_ngram: int = 3,
    min_ngram: int = 1,
    top_k: int = 200,
) -> Dict[str, Any]:
    """
    Extract and analyze recurring concepts from dataset captions.
    Filters out trigger words and phrases dominated by trigger words.
    """
    ds = _loaded_datasets.get(dataset_id)
    if not ds:
        raise ValueError(f"Dataset {dataset_id} not loaded")

    captions = [e.caption for e in ds.entries if e.caption]

    # First pass: detect trigger words
    # Collect all unigrams and check frequency
    all_words: Counter = Counter()
    for caption in captions:
        text = re.sub(r'[^\w\s\-]', ' ', caption.lower())
        for word in text.split():
            if word not in STOP_WORDS and len(word) > 2:
                all_words[word] += 1

    trigger_words = set()
    for word, count in all_words.items():
        if _is_likely_trigger_word(word, captions):
            trigger_words.add(word)

    # Second pass: extract ngrams, excluding trigger-word-dominated phrases
    phrase_to_indices: Dict[str, Set[int]] = defaultdict(set)

    for idx, entry in enumerate(ds.entries):
        if not entry.caption:
            continue
        ngrams = _extract_ngrams(entry.caption, min_n=min_ngram, max_n=max_ngram)
        for phrase in ngrams:
            words = phrase.split()
            # Skip if phrase is just a trigger word
            if len(words) == 1 and words[0] in trigger_words:
                continue
            # Skip if phrase is entirely trigger words + stop words
            meaningful = [w for w in words if w not in trigger_words and w not in STOP_WORDS]
            if not meaningful:
                continue
            # Skip ngrams that are "a <trigger>" or "<trigger> <common>"
            # i.e. where removing the trigger leaves only stop words or a single word
            non_trigger = [w for w in words if w not in trigger_words]
            if non_trigger != words:
                # Phrase contains a trigger word — only keep if the remaining
                # content is itself meaningful (2+ non-stop words)
                non_stop_non_trigger = [w for w in non_trigger if w not in STOP_WORDS]
                if len(non_stop_non_trigger) < 1:
                    continue
                # Rebuild phrase without trigger word
                phrase = ' '.join(non_trigger)
                if not phrase.strip() or phrase in STOP_WORDS:
                    continue

            phrase_to_indices[phrase].add(idx)

    # Filter by minimum frequency
    concepts = []
    for phrase, indices in phrase_to_indices.items():
        if len(indices) >= min_frequency:
            concepts.append(ConceptInfo(
                phrase=phrase,
                count=len(indices),
                image_indices=sorted(indices),
                category=_categorize_phrase(phrase),
            ))

    # Deduplicate: if "black purse" and "purse" both exist, and "purse" only
    # appears in images where "black purse" also appears, drop "purse"
    phrase_set = {c.phrase: c for c in concepts}
    to_remove = set()
    for c in concepts:
        words = c.phrase.split()
        if len(words) == 1:
            # Check if this unigram is always part of a longer phrase
            parent_phrases = [
                p for p in phrase_set.values()
                if len(p.phrase.split()) > 1 and c.phrase in p.phrase.split()
                and set(c.image_indices) <= set(p.image_indices)
            ]
            if parent_phrases:
                to_remove.add(c.phrase)

    concepts = [c for c in concepts if c.phrase not in to_remove]

    # Sort by frequency descending
    concepts.sort(key=lambda c: c.count, reverse=True)
    concepts = concepts[:top_k]

    # Co-occurrence matrix
    top_phrases = [c.phrase for c in concepts[:50]]
    cooccurrence: Dict[str, Dict[str, int]] = {}
    for i, p1 in enumerate(top_phrases):
        cooccurrence[p1] = {}
        for j, p2 in enumerate(top_phrases):
            if i == j:
                continue
            overlap = len(phrase_to_indices.get(p1, set()) & phrase_to_indices.get(p2, set()))
            if overlap > 0:
                cooccurrence[p1][p2] = overlap

    total_captions = len(captions)
    category_counts = Counter(c.category for c in concepts)

    return {
        "dataset_id": dataset_id,
        "total_images": ds.total_images,
        "total_with_captions": total_captions,
        "trigger_words_detected": sorted(trigger_words),
        "concepts": [c.to_dict() for c in concepts],
        "cooccurrence": cooccurrence,
        "category_distribution": dict(category_counts),
        "settings": {
            "min_frequency": min_frequency,
            "max_ngram": max_ngram,
            "min_ngram": min_ngram,
            "top_k": top_k,
        },
    }

def get_concept_images(
    dataset_id: str, phrase: str
) -> List[Dict[str, Any]]:
    """Get all images associated with a concept phrase."""
    ds = _loaded_datasets.get(dataset_id)
    if not ds:
        return []

    results = []
    lower_phrase = phrase.lower()
    for idx, entry in enumerate(ds.entries):
        if lower_phrase in entry.caption.lower():
            results.append({
                "index": idx,
                "filename": entry.filename,
                "caption": entry.caption,
                "width": entry.width,
                "height": entry.height,
            })
    return results


def find_similar_phrases(
    concepts: List[Dict[str, Any]], threshold: float = 0.6
) -> List[List[str]]:
    """
    Find groups of similar phrases that might be the same concept.
    Uses simple word overlap (Jaccard similarity).
    """
    phrases = [c["phrase"] for c in concepts]
    groups: List[List[str]] = []
    used: Set[int] = set()

    for i, p1 in enumerate(phrases):
        if i in used:
            continue
        words1 = set(p1.split())
        group = [p1]
        for j, p2 in enumerate(phrases):
            if j <= i or j in used:
                continue
            words2 = set(p2.split())
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            if union > 0 and intersection / union >= threshold:
                group.append(p2)
                used.add(j)
        if len(group) > 1:
            groups.append(group)
            used.add(i)

    return groups
