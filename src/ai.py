"""AI and search functionality for the semantic note search application."""

import asyncio
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import torch
from sentence_transformers import SentenceTransformer, util
import spacy

from config import (
    MODEL_NAME,
    QUERY_INSTRUCTION,
    SCORE_THRESHOLD,
    WIKILINK_SCORE_THRESHOLD,
    MAX_RESULTS,
    get_cache_file,
)


model: Optional[SentenceTransformer] = None
cache: Optional[Dict[str, Tuple[str, torch.Tensor]]] = None


def get_gpu_status() -> Dict[str, Any]:
    """Check GPU availability and return status info."""
    status_info = {
        "available": False,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": 0,
        "device_name": None,
        "device": "cpu",
    }

    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            status_info["available"] = True
            status_info["device_count"] = torch.cuda.device_count()
            status_info["device_name"] = (
                torch.cuda.get_device_name(0)
                if status_info["device_count"] > 0
                else None
            )
            status_info["device"] = "cuda"
        else:
            # Check for other backends like MPS (Apple Silicon) or DirectML
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                status_info["available"] = True
                status_info["device"] = "mps"
                status_info["device_name"] = "Apple Silicon GPU"
            elif (
                hasattr(torch.backends, "opencl")
                and torch.backends.opencl.is_available()
            ):
                status_info["available"] = True
                status_info["device"] = "opencl"
                status_info["device_name"] = "OpenCL GPU"
    except Exception:
        # If any error occurs during detection, fall back to CPU
        pass

    return status_info


def load_model() -> Optional[SentenceTransformer]:
    """Load the sentence transformer model."""
    global model

    # Get GPU status and determine best device
    gpu_status = get_gpu_status()

    if gpu_status["available"]:
        device = gpu_status["device"]
        print(f"Using {device} for model encoding ({gpu_status['device_name']})")

        # For CUDA, set the device to the first available GPU
        if device == "cuda" and gpu_status["device_count"] > 0:
            torch.cuda.set_device(0)
    else:
        device = "cpu"
        print("Using CPU for model encoding (no GPU detected)")

    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
        return model
    except Exception as e:
        print(f"Failed to load model on {device}, falling back to CPU: {e}")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device="cpu")
        return model


def get_all_notes(directory: Path) -> List[Path]:
    """Get all note files from the directory."""
    if not directory.is_dir():
        raise SystemExit(1)

    notes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in {".txt", ".md"}:
                notes.append(Path(root) / file)
    return notes


async def build_cache(
    notes: List[Path],
    model: SentenceTransformer,
    cache_file: Path,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Tuple[str, torch.Tensor]]:
    """Build the search cache for notes."""
    global cache
    cache = {}
    processed = 0

    for note_path in notes:
        try:
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                embedding = model.encode(
                    QUERY_INSTRUCTION + content,
                    convert_to_tensor=True,
                    device=next(model.parameters()).device,
                )
                cache[str(note_path)] = (content, embedding)
        except Exception:
            # Skip problematic notes but continue processing
            pass

        processed += 1

        # Update progress and yield control to UI
        if progress_callback:
            progress = int((processed / len(notes)) * 100)
            progress_callback(progress, f"Processed {processed}/{len(notes)} notes")
        await asyncio.sleep(0.01)  # Allow UI to update

    # Move tensors to CPU before saving (CUDA tensors can't be pickled)
    cpu_cache = {}
    for path, (content, embedding) in cache.items():
        cpu_cache[path] = (
            content,
            embedding.cpu() if hasattr(embedding, "cpu") else embedding,
        )

    with open(cache_file, "wb") as f:
        pickle.dump(cpu_cache, f)
    return cache


def load_cache(cache_file: Path) -> Dict[str, Tuple[str, torch.Tensor]]:
    """Load the search cache from file."""
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


async def update_cache_for_new_notes(
    model: SentenceTransformer,
    existing_cache: Dict[str, Tuple[str, torch.Tensor]],
    new_notes: Set[str],
    progress_callback: Optional[Any] = None,
) -> Dict[str, Tuple[str, torch.Tensor]]:
    """Update cache with new notes."""
    global cache
    total = len(new_notes)
    for i, note_path in enumerate(new_notes):
        try:
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                embedding = model.encode(
                    QUERY_INSTRUCTION + content,
                    convert_to_tensor=True,
                    device=next(model.parameters()).device,
                )
                cache[str(note_path)] = (content, embedding)
                existing_cache[str(note_path)] = (content, embedding)
        except Exception:
            # Skip problematic notes but continue processing
            pass

        if progress_callback:
            progress = int(((i + 1) / total) * 100) if total > 0 else 100
            progress_callback(progress, f"Processing new notes: {i + 1}/{total}")
        await asyncio.sleep(0.01)
    return existing_cache


def remove_deleted_notes_from_cache(
    existing_cache: Dict[str, Tuple[str, torch.Tensor]], removed_notes: Set[str]
) -> Dict[str, Tuple[str, torch.Tensor]]:
    """Remove deleted notes from cache."""
    global cache
    for path in removed_notes:
        cache.pop(path, None)
        existing_cache.pop(path, None)
    return existing_cache


async def load_or_build_cache(
    model: Optional[SentenceTransformer],
    current_note_paths: Set[str],
    notes_dir: Path,
    force_rebuild: bool = False,
    build_func: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
) -> Tuple[Dict[str, Tuple[str, torch.Tensor]], str]:
    """Load or build cache and return status information."""
    global cache
    if build_func is None:
        build_func = build_cache

    cache_file = get_cache_file(notes_dir)
    if force_rebuild:
        if cache_file.exists():
            cache_file.unlink()

    cache = load_cache(cache_file)

    # Move tensors to correct device after loading
    if cache and model:
        try:
            device = next(model.parameters()).device
            for path, (content, embedding) in cache.items():
                if hasattr(embedding, "to"):
                    cache[path] = (content, embedding.to(device))
        except Exception:
            # If device placement fails, keep tensors on CPU
            pass

    if not cache:
        notes = [Path(p) for p in current_note_paths]
        cache = await build_func(notes, model, cache_file, progress_callback)
        return cache, "Indexed notes"

    cached_note_paths = set(cache.keys())
    new_notes = current_note_paths - cached_note_paths
    removed_notes = cached_note_paths - current_note_paths

    if not new_notes and not removed_notes and not force_rebuild:
        return cache, "Cache up to date"
    else:
        status_parts = []
        if force_rebuild:
            notes = [Path(p) for p in current_note_paths]
            cache = await build_func(notes, model, cache_file, progress_callback)
            status_parts.append("reindexed")
        else:
            if new_notes:
                cache = await update_cache_for_new_notes(
                    model, cache, new_notes, progress_callback
                )
                status_parts.append(f"added {len(new_notes)} new notes")

            if removed_notes:
                cache = remove_deleted_notes_from_cache(cache, removed_notes)
                status_parts.append(f"removed {len(removed_notes)} deleted notes")

        # Move tensors to CPU before saving (CUDA tensors can't be pickled)
        cpu_cache = {}
        for path, (content, embedding) in cache.items():
            cpu_cache[path] = (
                content,
                embedding.cpu() if hasattr(embedding, "cpu") else embedding,
            )

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cpu_cache, f)
        except Exception:
            # If saving fails, continue but log the error
            pass

        return cache, f"Updated cache: {', '.join(status_parts)}"


def load_spacy_model() -> Optional[Any]:
    """Load spaCy model with fallback handling."""
    if spacy is None:
        print("Warning: spaCy not available. Install with: pip install spacy")
        return None

    try:
        # Try to load the English model
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy model loaded successfully")
        return nlp
    except OSError:
        print("spaCy English model not found. Attempting to download...")
        try:
            from spacy.cli import download

            download("en_core_web_sm")
            print("✓ spaCy model downloaded successfully")
            # Now try loading again
            nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded successfully")
            return nlp
        except Exception as e:
            print(f"Error downloading spaCy model: {e}")
            return None
    except Exception as e:
        print(f"Warning: Could not load spaCy model: {e}")
        return None


def extract_noun_phrases(text: str, nlp: Optional[Any]) -> List[str]:
    """Extract noun phrases from text using spaCy."""
    if not nlp:
        return []

    try:
        doc = nlp(text)
        noun_phrases = []

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if len(phrase) > 2:  # Filter out very short phrases
                noun_phrases.append(phrase)

        return noun_phrases
    except Exception as e:
        print(f"Warning: Could not extract noun phrases: {e}")
        return []


def extract_verb_phrases(text: str, nlp: Optional[Any]) -> List[str]:
    """Extract verb phrases from text using spaCy, excluding trivial verbs."""
    if not nlp:
        return []

    try:
        doc = nlp(text)
        verb_phrases = []
        trivial_verbs = {
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "do",
            "does",
            "did",
            "have",
            "has",
            "had",
            "will",
            "would",
            "can",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "get",
            "gets",
            "got",
            "go",
            "goes",
            "went",
            "come",
            "comes",
            "came",
            "make",
            "makes",
            "made",
            "take",
            "takes",
            "took",
            "give",
            "gives",
            "gave",
            "see",
            "sees",
            "saw",
            "know",
            "knows",
            "knew",
        }

        for token in doc:
            if token.pos_ == "VERB" and token.text.lower() not in trivial_verbs:
                # Get the verb phrase including auxiliaries and subtrees
                phrase_tokens = [token]

                # Add auxiliary verbs
                for child in token.children:
                    if child.dep_ in ["aux", "auxpass", "neg"]:
                        phrase_tokens.append(child)

                # Add the verb phrase
                phrase = " ".join([t.text for t in phrase_tokens]).strip()
                if len(phrase) > 2:
                    verb_phrases.append(phrase)

        return verb_phrases
    except Exception as e:
        print(f"Warning: Could not extract verb phrases: {e}")
        return []


def combine_and_deduplicate_candidates(
    noun_phrases: List[str], verb_phrases: List[str]
) -> List[str]:
    """Combine noun and verb phrases and remove duplicates."""
    all_candidates = noun_phrases + verb_phrases

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []

    for phrase in all_candidates:
        phrase_lower = phrase.lower()
        if phrase_lower not in seen:
            seen.add(phrase_lower)
            unique_candidates.append(phrase)

    return unique_candidates


def filter_text_for_candidates(text: str) -> str:
    """Filter text to exclude headings, yaml frontmatter, and existing wikilinks."""
    lines = text.split("\n")
    filtered_lines = []
    in_yaml = False

    for line in lines:
        # Skip yaml frontmatter
        if line.strip() == "---":
            in_yaml = not in_yaml
            continue
        if in_yaml:
            continue

        # Skip headings (lines starting with #)
        if line.strip().startswith("#"):
            continue

        # Skip markdown list items (lines starting with - )
        if line.strip().startswith("- "):
            continue

        # Remove URLs from the line
        import re

        line = re.sub(r"https?://\S+", "", line)

        # Remove existing wikilinks from the line
        line = re.sub(r"\[\[[^\]]*\]\]", "", line)

        # Only add non-empty lines
        if line.strip():
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def analyze_text_for_wikilinks(
    text: str,
    model: SentenceTransformer | None,
    cache: Dict[str, Tuple[str, torch.Tensor]],
    nlp: Optional[Any] = None,
    source_note_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Analyze text for wikilink candidates and return suggestions."""
    if model is None:
        raise Exception("no model")
    if not text.strip() or not model or not cache:
        return []

    # Filter text to ignore headings, yaml frontmatter, and existing links
    filtered_text = filter_text_for_candidates(text)

    # Extract candidates using NLP
    noun_phrases = extract_noun_phrases(filtered_text, nlp)
    verb_phrases = extract_verb_phrases(filtered_text, nlp)
    candidates = combine_and_deduplicate_candidates(noun_phrases, verb_phrases)

    if not candidates:
        return []

    try:
        # Embed all candidates
        candidate_embeddings = model.encode(
            candidates, convert_to_tensor=True, device=next(model.parameters()).device
        )
        note_paths = list(cache.keys())
        note_embeddings_list = [data[1] for data in cache.values()]
        note_embeddings_tensor = torch.stack(note_embeddings_list)

        # Calculate similarities
        similarities = util.cos_sim(candidate_embeddings, note_embeddings_tensor)

        # Process results
        wikilink_suggestions = []
        for i, candidate in enumerate(candidates):
            candidate_similarities = similarities[i]

            # Find the best match for this candidate
            best_match_idx = int(torch.argmax(candidate_similarities))
            best_score = float(candidate_similarities[best_match_idx])

            if best_score >= WIKILINK_SCORE_THRESHOLD:
                best_note_path = note_paths[best_match_idx]
                best_note_content = cache[best_note_path][0]

                # Prevent self-linking: skip if target is the same as source
                if (
                    source_note_path
                    and Path(best_note_path).name == Path(source_note_path).name
                ):
                    continue

                # Format as wikilink suggestion
                filename = Path(best_note_path).name
                wikilink_suggestion = {
                    "candidate": candidate,
                    "filename": filename,
                    "score": best_score,
                    "wikilink": f"[[{filename}|{candidate}]]",
                    "note_content": best_note_content[:200] + "..."
                    if len(best_note_content) > 200
                    else best_note_content,
                }
                wikilink_suggestions.append(wikilink_suggestion)

        # Sort by similarity score
        wikilink_suggestions.sort(key=lambda x: x["score"], reverse=True)

        return wikilink_suggestions

    except Exception:
        return []


def search(
    query: str,
    model: SentenceTransformer | None,
    cache: Dict[str, Tuple[str, torch.Tensor]],
    score_threshold: float = SCORE_THRESHOLD,
    max_results: int = MAX_RESULTS,
) -> List[Tuple[float, str, str]]:
    """Search for notes matching the query."""
    if model is None:
        raise Exception("no model")
    if not query.strip():
        return []

    query_embedding = model.encode(
        QUERY_INSTRUCTION + query,
        convert_to_tensor=True,
        device=next(model.parameters()).device,
    )

    note_paths = list(cache.keys())
    note_embeddings_list = [data[1] for data in cache.values()]

    if not note_embeddings_list:
        return []

    note_embeddings_tensor = torch.stack(note_embeddings_list)

    cosine_scores = util.cos_sim(query_embedding, note_embeddings_tensor)

    all_results = []
    for idx, score in enumerate(cosine_scores[0]):
        note_path = note_paths[idx]
        content = cache[note_path][0]
        all_results.append((score.item(), note_path, content))

    all_results.sort(key=lambda x: x[0], reverse=True)

    # Filter by threshold and limit results
    results = [r for r in all_results if r[0] >= score_threshold][:max_results]

    return results
