"""AI and search functionality for the semantic note search application."""

import asyncio
import os
import pickle
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from config import (
    MODEL_NAME, QUERY_INSTRUCTION, SCORE_THRESHOLD, WIKILINK_SCORE_THRESHOLD,
    MAX_RESULTS, get_cache_file
)

try:
    import spacy
except ImportError:
    spacy = None

model = None
cache = None


def load_model():
    """Load the sentence transformer model."""
    global model
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("Model loaded successfully.")
    return model


def get_all_notes(directory):
    """Get all note files from the directory."""
    if not directory.is_dir():
        print(f"Error: Notes directory not found at '{directory}'")
        print(
            "Please create it or use a different directory: python main.py /path/to/notes"
        )
        raise SystemExit(1)

    print(f"Scanning for notes in '{directory}'...")
    notes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in {".txt", ".md"}:
                notes.append(Path(root) / file)
    print(f"Found {len(notes)} notes.")
    return notes


async def build_cache(notes, model, cache_file, progress_callback=None):
    """Build the search cache for notes."""
    print(f"Building cache for {len(notes)} notes... This may take a while for the first time.")
    cache = {}
    processed = 0

    for note_path in notes:
        try:
            print(f"Processing note {processed + 1}/{len(notes)}: {note_path.name}")
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                print(f"  Generating embedding for {len(content)} characters...")
                embedding = model.encode(QUERY_INSTRUCTION + content, convert_to_tensor=True)
                cache[str(note_path)] = (content, embedding)
                print(f"  ✓ Processed {note_path.name}")
            else:
                print(f"  ⚠ Skipped empty note: {note_path.name}")
        except Exception as e:
            print(f"  ✗ Error processing {note_path}: {e}")

        processed += 1

        # Update progress and yield control to UI
        if progress_callback:
            progress = int((processed / len(notes)) * 100)
            progress_callback(progress, f"Processed {processed}/{len(notes)} notes")
        await asyncio.sleep(0.01)  # Allow UI to update

    print(f"Cache built with {len(cache)} notes. Saving to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    print("Cache saved successfully.")
    return cache


def load_cache(cache_file):
    """Load the search cache from file."""
    print(f"Loading cache from {cache_file}")
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Index file not found, will index notes.")
        return {}
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}


async def update_cache_for_new_notes(model, cache, new_notes, progress_callback=None):
    """Update cache with new notes."""
    total = len(new_notes)
    for i, note_path in enumerate(new_notes):
        try:
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                embedding = model.encode(QUERY_INSTRUCTION + content, convert_to_tensor=True)
                cache[str(note_path)] = (content, embedding)
        except Exception as e:
            print(f"Warning: Could not process new note {note_path}: {e}")

        if progress_callback:
            progress = int(((i + 1) / total) * 100) if total > 0 else 100
            progress_callback(progress, f"Processing new notes: {i + 1}/{total}")
        await asyncio.sleep(0.01)
    return cache


def remove_deleted_notes_from_cache(cache, removed_notes):
    """Remove deleted notes from cache."""
    for path in removed_notes:
        cache.pop(path, None)
    return cache


async def load_or_build_cache(model, current_note_paths, notes_dir, force_rebuild=False, build_func=None, progress_callback=None):
    """Load or build cache and return status information."""
    if build_func is None:
        build_func = build_cache

    cache_file = get_cache_file(notes_dir)
    if force_rebuild:
        print("Force reindex requested - removing existing index...")
        if cache_file.exists():
            cache_file.unlink()
            print("Existing index removed.")
        else:
            print("No existing index file found to remove.")

    print(f"Loading existing index from {cache_file}...")
    cache = load_cache(cache_file)

    if not cache:
        print("No existing index found. Indexing notes from scratch...")
        notes = [Path(p) for p in current_note_paths]
        print(f"Processing {len(notes)} notes for initial cache...")
        return await build_func(notes, model, cache_file, progress_callback), "Indexed notes"

    print(f"Found existing index with {len(cache)} notes")
    cached_note_paths = set(cache.keys())
    print(f"Comparing {len(current_note_paths)} current notes with {len(cached_note_paths)} cached notes...")

    new_notes = current_note_paths - cached_note_paths
    removed_notes = cached_note_paths - current_note_paths

    print(f"Found {len(new_notes)} new notes and {len(removed_notes)} removed notes")

    if not new_notes and not removed_notes and not force_rebuild:
        print("Index is up to date - no changes needed.")
        return cache, "Cache up to date"
    else:
        status_parts = []
        if force_rebuild:
            print("Force reindexing entire collection...")
            notes = [Path(p) for p in current_note_paths]
            cache = await build_func(notes, model, cache_file, progress_callback)
            status_parts.append("reindexed")
        else:
            if new_notes:
                print(f"Adding {len(new_notes)} new notes to cache...")
                cache = await update_cache_for_new_notes(model, cache, new_notes, progress_callback)
                status_parts.append(f"added {len(new_notes)} new notes")

            if removed_notes:
                print(f"Removing {len(removed_notes)} deleted notes from cache...")
                remove_deleted_notes_from_cache(cache, removed_notes)
                status_parts.append(f"removed {len(removed_notes)} deleted notes")

        print(f"Saving updated cache with {len(cache)} total notes...")
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        print("Cache saved successfully.")

        return cache, f"Updated cache: {', '.join(status_parts)}"


def load_spacy_model():
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
        print("Warning: spaCy English model not found.")
        print("Please install with: python -m spacy download en_core_web_sm")
        return None
    except Exception as e:
        print(f"Warning: Could not load spaCy model: {e}")
        return None


def extract_noun_phrases(text, nlp):
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


def extract_verb_phrases(text, nlp):
    """Extract verb phrases from text using spaCy, excluding trivial verbs."""
    if not nlp:
        return []

    try:
        doc = nlp(text)
        verb_phrases = []
        trivial_verbs = {"is", "are", "was", "were", "be", "been", "being",
                        "do", "does", "did", "have", "has", "had", "will",
                        "would", "can", "could", "should", "may", "might",
                        "must", "shall", "get", "gets", "got", "go", "goes",
                        "went", "come", "comes", "came", "make", "makes",
                        "made", "take", "takes", "took", "give", "gives",
                        "gave", "see", "sees", "saw", "know", "knows", "knew"}

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


def combine_and_deduplicate_candidates(noun_phrases, verb_phrases):
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


def filter_text_for_candidates(text):
    """Filter text to exclude headings, yaml frontmatter, and existing wikilinks."""
    lines = text.split('\n')
    filtered_lines = []
    in_yaml = False

    for line in lines:
        # Skip yaml frontmatter
        if line.strip() == '---':
            in_yaml = not in_yaml
            continue
        if in_yaml:
            continue

        # Skip headings (lines starting with #)
        if line.strip().startswith('#'):
            continue

        # Remove existing wikilinks from the line
        import re
        line = re.sub(r'\[\[[^\]]*\]\]', '', line)

        # Only add non-empty lines
        if line.strip():
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def analyze_text_for_wikilinks(text, model, cache, nlp=None, source_note_path=None):
    """Analyze text for wikilink candidates and return suggestions."""
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

    print(f"Analyzing {len(candidates)} candidates...")

    try:
        # Embed all candidates
        candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
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
                if source_note_path and Path(best_note_path).name == Path(source_note_path).name:
                    continue

                # Format as wikilink suggestion
                filename = Path(best_note_path).name
                wikilink_suggestion = {
                    'candidate': candidate,
                    'filename': filename,
                    'score': best_score,
                    'wikilink': f"[[{filename}|{candidate}]]",
                    'note_content': best_note_content[:200] + "..." if len(best_note_content) > 200 else best_note_content
                }
                wikilink_suggestions.append(wikilink_suggestion)

        # Sort by similarity score
        wikilink_suggestions.sort(key=lambda x: x['score'], reverse=True)

        print(f"Found {len(wikilink_suggestions)} wikilink suggestions")
        return wikilink_suggestions

    except Exception as e:
        print(f"Error analyzing text for wikilinks: {e}")
        return []


def search(query, model, cache, score_threshold=SCORE_THRESHOLD, max_results=MAX_RESULTS):
    """Search for notes matching the query."""
    if not query.strip():
        return []

    print(f"Searching for: '{query}'")
    query_embedding = model.encode(QUERY_INSTRUCTION + query, convert_to_tensor=True)

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