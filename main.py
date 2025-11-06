#!/usr/bin/env python3

import os
import pickle
import sys
import asyncio
from pathlib import Path
from typing import cast

from textual.containers import Container

import torch
from sentence_transformers import SentenceTransformer, util
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, TextArea, Button, ProgressBar, Label
from textual.binding import Binding
from textual.screen import Screen, ModalScreen

try:
    import spacy
except ImportError:
    spacy = None

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Semantic note search using Sentence Transformers"
    )
    parser.add_argument(
        "notes_dir",
        nargs="?",
        default="test_data",
        help="Directory containing note files (default: ./test_data)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with dummy data (no AI loading)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild the search cache from scratch",
    )
    try:
        return parser.parse_args()
    except SystemExit:
        # During testing, pytest may pass unknown arguments, so return defaults
        return argparse.Namespace(notes_dir="test_data", test_mode=False, rebuild_cache=False)


def get_notes_dir():
    args = parse_arguments()
    return Path(args.notes_dir)


CACHE_FILE = Path(__file__).parent / "cache.pkl"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
NOTE_EXTENSIONS = {".txt", ".md"}
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SCORE_THRESHOLD = 0.52
WIKILINK_SCORE_THRESHOLD = 0.65
MIN_RESULTS = 12
MAX_RESULTS = 16

# Application modes
MODE_SEARCH = "search"
MODE_ANALYZE = "analyze"

model = None
cache = None


def load_model():
    global model
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("Model loaded successfully.")
    return model


def get_all_notes(directory):
    if not directory.is_dir():
        print(f"Error: Notes directory not found at '{directory}'")
        print(
            "Please create it or use a different directory: python main.py /path/to/notes"
        )
        sys.exit(1)

    print(f"Scanning for notes in '{directory}'...")
    notes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in NOTE_EXTENSIONS:
                notes.append(Path(root) / file)
    print(f"Found {len(notes)} notes.")
    return notes


def build_cache(notes, model):
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

    print(f"Cache built with {len(cache)} notes. Saving to {CACHE_FILE}...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print("Cache saved successfully.")
    return cache



def load_cache():
    print(f"Loading cache from {CACHE_FILE}")
    try:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Index file not found, will index notes.")
        return {}
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}


def update_cache_for_new_notes(model, cache, new_notes):
    for note_path in new_notes:
        try:
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                embedding = model.encode(QUERY_INSTRUCTION + content, convert_to_tensor=True)
                cache[str(note_path)] = (content, embedding)
        except Exception as e:
            print(f"Warning: Could not process new note {note_path}: {e}")
    return cache


def remove_deleted_notes_from_cache(cache, removed_notes):
    for path in removed_notes:
        cache.pop(path, None)
    return cache


def load_or_build_cache(model, current_note_paths, force_rebuild=False, build_func=None):
    """Load or build cache and return status information."""
    if build_func is None:
        build_func = build_cache

    if force_rebuild:
        print("Force reindex requested - removing existing index...")
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            print("Existing index removed.")
        else:
            print("No existing index file found to remove.")

    print(f"Loading existing index from {CACHE_FILE}...")
    cache = load_cache()

    if not cache:
        print("No existing index found. Indexing notes from scratch...")
        notes = [Path(p) for p in current_note_paths]
        print(f"Processing {len(notes)} notes for initial cache...")
        return build_func(notes, model), "Indexed notes"

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
            cache = build_func(notes, model)
            status_parts.append("reindexed")
        else:
            if new_notes:
                print(f"Adding {len(new_notes)} new notes to cache...")
                update_cache_for_new_notes(model, cache, new_notes)
                status_parts.append(f"added {len(new_notes)} new notes")

            if removed_notes:
                print(f"Removing {len(removed_notes)} deleted notes from cache...")
                remove_deleted_notes_from_cache(cache, removed_notes)
                status_parts.append(f"removed {len(removed_notes)} deleted notes")

        print(f"Saving updated cache with {len(cache)} total notes...")
        with open(CACHE_FILE, "wb") as f:
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


def analyze_text_for_wikilinks(text, model, cache, nlp=None):
    """Analyze text for wikilink candidates and return suggestions."""
    if not text.strip() or not model or not cache:
        return []
    
    # Extract candidates using NLP
    noun_phrases = extract_noun_phrases(text, nlp)
    verb_phrases = extract_verb_phrases(text, nlp)
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


def search(
    query, model, cache, score_threshold=SCORE_THRESHOLD, max_results=MAX_RESULTS
):
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


class CustomInput(Input):
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all"),
        Binding("ctrl+backspace", "delete_left_word", "Delete previous word"),
        Binding("ctrl+delete", "delete_right_word", "Delete next word"),
    ]
    cursor_blink = True


class LoadingScreen(Screen):
    """Loading screen shown during app initialization."""

    def compose(self) -> ComposeResult:
        with Container(id="loading-screen"):
            with Vertical(id="loading-content"):
                # App title/logo
                yield Label("🔍 Semantic Note Search", id="loading-title")
                yield Label("Initializing your AI-powered note search...", id="loading-subtitle")

                # Progress section
                with Vertical(id="loading-progress-section"):
                    yield Label("Loading components...", id="loading-status")
                    yield ProgressBar(id="loading-progress", total=100)
                    yield RichLog(id="loading-log", markup=True, auto_scroll=False)

    def update_loading_log(self, message, clear_first=False):
        """Update the loading log with a message."""
        log = self.query_one("#loading-log", RichLog)
        if clear_first:
            log.clear()
        log.write(message)

    def update_status(self, status: str):
        """Update the loading status text."""
        status_label = self.query_one("#loading-status", Label)
        status_label.update(status)

    def update_progress(self, progress: int):
        """Update the loading progress bar."""
        progress_bar = self.query_one("#loading-progress", ProgressBar)
        progress_bar.progress = progress


class SearchScreen(Screen):
    """Main search interface screen."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="main-container"):
            # Search Mode Layout (2-pane)
            with Horizontal(classes="content-container", id="search-layout"):
                with Vertical(classes="search-results-panel"):
                    yield RichLog(id="results", auto_scroll=False, markup=True)
                with Vertical(classes="search-preview-panel"):
                    yield TextArea(
                        id="preview",
                        read_only=True,
                        language="markdown",
                        theme="dracula",
                    )

            with Horizontal(classes="search-container"):
                yield CustomInput(placeholder="Enter search query...", id="search-input")
                yield Button("Analyze", id="mode-btn", classes="mode-button")

    def on_mount(self):
        """Focus the search input when the screen mounts."""
        self.call_later(self.focus_input)

    def focus_input(self):
        """Focus the search input field."""
        try:
            search_input = self.query_one("#search-input", Input)
            search_input.focus()
        except:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission in search screen."""
        app = cast(SearchApp, self.app)
        if app.loading:
            return

        query = event.value.strip()
        if query:
            if app.app_mode == MODE_SEARCH:
                app.perform_search(query)
            else:
                app.analyze_text(query)
        else:
            app.clear_results()

        self.focus_input()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes in search screen."""
        app = cast(SearchApp, self.app)
        if app.loading:
            return

        query = event.value.strip()
        if query and len(query) >= 2:
            if app.app_mode == MODE_SEARCH:
                if app.test_mode:
                    # In test mode, perform search immediately without debouncing
                    app.perform_search(query)
                else:
                    app.debounce_search(query)
            else:
                if app.test_mode:
                    app.analyze_text(query)
                else:
                    app.debounce_analysis(query)
        elif not query:
            app.clear_results()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in search screen."""
        app = cast(SearchApp, self.app)
        if app.loading:
            return

        if event.button.id == "mode-btn":
            # This is the analyze button in search mode
            app.call_later(app.toggle_app_mode)


class AnalyzeScreen(Screen):
    """Analyze mode interface screen."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="main-container"):
            # Analyze Mode Layout (no search bar, different UI)
            with Vertical(classes="analyze-container", id="analyze-layout"):
                # Progress section
                with Vertical(id="analyze-progress-section"):
                    yield Label("Scan Progress", id="progress-label")
                    yield ProgressBar(id="analyze-progress", total=100)
                    yield Label("Ready to scan...", id="progress-status")

                # Main content
                with Horizontal(classes="analyze-main-content"):
                    with Vertical(classes="analyze-suggestions-panel"):
                        yield RichLog(id="analyze-results", auto_scroll=False, markup=True)
                    with Vertical(classes="analyze-preview-panel"):
                        yield TextArea(
                            id="analyze-preview",
                            read_only=True,
                            language="markdown",
                            theme="dracula",
                        )

    def update_progress(self, percentage: int, status: str):
        """Update the progress bar and status text."""
        try:
            progress_bar = self.query_one("#analyze-progress", ProgressBar)
            progress_bar.progress = percentage

            status_label = self.query_one("#progress-status", Label)
            status_label.update(status)
        except Exception as e:
            print(f"Error updating progress: {e}")

    def display_all_analysis_results(self) -> None:
        """Display ALL wikilink suggestions from all notes."""
        try:
            app = cast(SearchApp, self.app)
            results_log = self.query_one("#analyze-results", RichLog)
            results_log.clear()

            if not app.all_analysis_suggestions:
                results_log.write("No wikilink suggestions found")
                return

            results_log.write(f"[bold #cd5c5c]Wikilink Suggestions for All Notes[/bold #cd5c5c]")
            results_log.write("")
            results_log.write(f"Found {len(app.all_analysis_suggestions)} suggestions")
            results_log.write("")
            results_log.write("[dim]Navigate with ↑↓ arrows[/dim]")
            results_log.write("")

            # Display ALL wikilink suggestions
            for i, suggestion in enumerate(app.all_analysis_suggestions):
                score = suggestion['score']
                source_note = suggestion['source_note']
                wikilink = suggestion['wikilink']
                candidate = suggestion['candidate']

                is_selected = i == app.selected_suggestion_index

                if is_selected:
                    results_log.write(f"[bold #f5dede on #5d2828]{score:.3f}  {wikilink}[/]")
                    results_log.write(f"[#f5dede on #5d2828]   From: {source_note}[/]")
                    results_log.write(f"[#f5dede on #5d2828]   Candidate: {candidate}[/]")
                else:
                    results_log.write(f"[#cd5c5c]{score:.3f}  {wikilink}")
                    results_log.write(f"[dim]   From: {source_note}[/dim]")
                    results_log.write(f"[dim]   Candidate: {candidate}[/dim]")

                results_log.write("")

        except Exception as e:
            print(f"Error displaying all analysis results: {e}")

    def display_analysis_preview(self, index: int) -> None:
        """Display preview for selected wikilink suggestion."""
        try:
            app = cast(SearchApp, self.app)
            preview_area = self.query_one("#analyze-preview", TextArea)
            preview_area.clear()

            if not (0 <= index < len(app.all_analysis_suggestions)):
                preview_area.load_text("No suggestion selected")
                return

            suggestion = app.all_analysis_suggestions[index]

            # Create detailed preview content
            preview_content = f"Wikilink Suggestion Details\n"
            preview_content += f"=" * 30 + "\n\n"

            # Suggestion info
            preview_content += f"Suggested Wikilink: {suggestion['wikilink']}\n"
            preview_content += f"Candidate Phrase: {suggestion['candidate']}\n"
            preview_content += f"Similarity Score: {suggestion['score']:.4f}\n"
            preview_content += f"Source Note: {suggestion['source_note']}\n\n"

            # Context from source note
            preview_content += f"Context from Source Note:\n"
            preview_content += f"-" * 25 + "\n"
            preview_content += f"{suggestion['source_note_content']}\n\n"

            # Linked note preview
            linked_filename = suggestion['filename']
            preview_content += f"Linked Note Preview:\n"
            preview_content += f"-" * 20 + "\n"
            preview_content += f"File: {linked_filename}\n"
            preview_content += f"This note would be linked by the wikilink.\n"

            # Find and show actual linked note content
            global cache
            if cache:
                linked_note_path = None
                for path in cache.keys():
                    if Path(path).name == linked_filename:
                        linked_note_path = path
                        break

                if linked_note_path and linked_note_path in cache:
                    linked_content = cache[linked_note_path][0]
                    preview_content += f"\nActual content from {linked_filename}:\n"
                    preview_content += f"{linked_content[:300]}{'...' if len(linked_content) > 300 else ''}"
                else:
                    preview_content += f"\n(Note: {linked_filename} content not found in cache)"
            else:
                preview_content += f"\n(Note: Cache not available)"

            preview_area.load_text(preview_content)
            preview_area.scroll_home()

        except Exception as e:
            print(f"Error displaying analysis preview: {e}")


class ConfirmAnalyzeScreen(ModalScreen[bool]):
    """Confirmation screen for starting analyze mode scan."""

    def __init__(self, note_count: int):
        super().__init__()
        self.note_count = note_count

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(f"⚠️  Analyze Mode Scan Confirmation", id="confirm-title")
            yield Label("")
            yield Label(f"This will scan {self.note_count} notes for wikilink suggestions.", id="confirm-message")
            yield Label("This process may take several minutes depending on the number of notes.", id="confirm-warning")
            yield Label("")
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Start Scan", id="confirm-btn", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        elif event.button.id == "cancel-btn":
            self.dismiss(False)

    CSS = """
    #confirm-dialog {
        width: 70;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 2;
        align: center middle;
    }

    #confirm-title {
        text-align: center;
        color: $primary;
        text-style: bold;
    }

    #confirm-message, #confirm-warning {
        text-align: center;
        margin: 0 1;
    }

    #confirm-buttons {
        align: center middle;
        margin-top: 1;
    }

    #confirm-buttons Button {
        margin: 0 1;
        min-width: 12;
    }
    """


class SearchApp(App):
    """Main application for semantic note search."""

    CSS = """
    Screen { background: #1a0d0d; }
    .main-container { height: 100%; }
    .content-container { height: 1fr; }
    .search-results-panel {
        width: 1fr;
        border: solid #8b3a3a;
        margin: 0 1 0 1;
        background: #2d1414;
    }
    .search-preview-panel {
        width: 1fr;
        border: solid #cd5c5c;
        margin: 0 1 0 0;
        background: #2d1414;
    }
    .analyze-container {
        height: 1fr;
        background: #2d1414;
    }
    #analyze-progress-section {
        height: auto;
        background: #2d1414;
        border: solid #8b3a3a;
        margin: 0 1 1 1;
        padding: 1;
    }
    #progress-label {
        text-align: center;
        color: $primary;
        text-style: bold;
    }
    #analyze-progress {
        width: 1fr;
        margin: 0 2;
    }
    #progress-status {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    .analyze-main-content {
        height: 1fr;
    }
    .analyze-suggestions-panel {
        width: 1fr;
        border: solid #8b3a3a;
        margin: 0 1 0 1;
        background: #2d1414;
    }
    .analyze-preview-panel {
        height: 1fr;
        border: solid #cd5c5c;
        margin: 0 1 0 0;
        background: #2d1414;
    }
    .search-container {
        height: auto;
        background: #2d1414;
        padding: 0 1;
        align: center middle;
    }
    #search-input {
        width: 1fr;
        height: 3;
        min-height: 3;
        color: #f5dede;
        background: #2d1414;
        border: solid #8b3a3a;
        padding: 0 1;
    }
    #search-input:focus {
        border: solid #cd5c5c;
    }
    .mode-button {
        width: 12;
        min-width: 12;
        height: 3;
        margin: 0 0 0 1;
        color: #f5dede;
        background: #8b3a3a;
        border: solid #8b3a3a;
        text-align: center;
    }
    .mode-button:hover {
        background: #a14d4d;
        border: solid #cd5c5c;
    }
    .mode-button:focus {
        background: #a14d4d;
        border: solid #cd5c5c;
    }
    #results {
        height: 100%;
        scrollbar-size: 1 1;
        color: #f5dede;
        overflow-y: auto;
    }
    #preview {
        height: 100%;
        scrollbar-size: 1 1;
        color: #f5dede;
        background: #2d1414;
        overflow-y: auto;
    }
    .selected-result { background: #5d2828; }
    #loading-screen {
        width: 100%;
        height: 100%;
        background: #1a0d0d;
        align: center middle;
    }
    #loading-content {
        width: 70;
        height: auto;
        background: #2d1414;
        border: thick #cd5c5c;
        padding: 3;
        align: center middle;
    }
    #loading-title {
        text-align: center;
        color: #f5dede;
        text-style: bold;
        margin-bottom: 1;
    }
    #loading-subtitle {
        text-align: center;
        color: #cd5c5c;
        margin-bottom: 2;
    }
    #loading-progress-section {
        width: 100%;
        align: center middle;
    }
    #loading-status {
        text-align: center;
        color: #f5dede;
        margin-bottom: 1;
    }
    #loading-progress {
        width: 50;
        margin: 1 0;
    }
    #loading-log {
        width: 100%;
        height: 8;
        background: #1a0d0d;
        border: solid #8b3a3a;
        margin-top: 2;
        color: #cd5c5c;
    }
    """

    SCREENS = {
        "loading": LoadingScreen,
        "search": SearchScreen,
        "analyze": AnalyzeScreen,
    }

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.current_results = []
        self.current_suggestions = []
        self.current_analysis_results = []
        self.selected_index = 0
        self.selected_suggestion_index = 0
        self.loading = True
        self._search_timer = None
        self.app_mode = MODE_SEARCH
        self.all_analysis_suggestions = []  # Store ALL suggestions for analyze mode
        self.test_mode = False

    def compose(self) -> ComposeResult:
        # The app uses separate screens now, so compose is minimal
        return
        yield  # Make it a generator function

    def on_mount(self) -> None:
        # Start with loading screen
        self.push_screen("loading")
        # Start initialization immediately with a small delay to ensure UI is ready
        self.set_timer(0.05, self.start_async_initialization)

    def start_async_initialization(self) -> None:
        """Start the async initialization process."""
        # Use Textual's call_later to run async code
        self.call_later(self.initialize_app_async)



    async def initialize_app_async(self) -> None:
        """Initialize the application asynchronously."""
        global model, cache

        loading_screen = cast(LoadingScreen, self.screen)
        loading_screen.query_one("#loading-log", RichLog).clear()
        loading_screen.update_progress(0)

        try:
            # Parse arguments for both modes
            args = parse_arguments()

            # Step 0: Welcome message
            loading_screen.update_status("Initializing...")
            loading_screen.update_loading_log("[bold #cd5c5c]Welcome to Semantic Note Search![/bold #cd5c5c]", True)
            if self.test_mode:
                loading_screen.update_loading_log("[yellow]TEST MODE ENABLED[/yellow]")
            loading_screen.update_loading_log("")
            await self.force_ui_update()

            if not self.test_mode:
                # Step 1: Load model
                loading_screen.update_status("Loading AI model...")
                loading_screen.update_loading_log("[#f5dede]Loading sentence transformer model...[/#f5dede]")
                loading_screen.update_progress(10)
                await self.force_ui_update()

                model = load_model()
                loading_screen.update_progress(40)

                loading_screen.update_loading_log("[#90ee90]✓ Model loaded successfully[/#90ee90]", True)
                await self.force_ui_update()

                # Step 2: Scan for notes
                loading_screen.update_status("Scanning notes...")
                loading_screen.update_loading_log("[#f5dede]Scanning for notes...[/#f5dede]")
                loading_screen.update_progress(50)
                await self.force_ui_update()

                current_notes = get_all_notes(get_notes_dir())
                current_note_paths = {str(p) for p in current_notes}
                loading_screen.update_progress(70)

                loading_screen.update_loading_log(
                    f"[#90ee90]✓ Found {len(current_notes)} notes in {get_notes_dir()}[/#90ee90]",
                    True,
                )
                await self.force_ui_update()

                # Step 3: Build/rebuild cache
                loading_screen.update_status("Indexing notes...")
                loading_screen.update_loading_log("[#f5dede]Indexing notes...[/#f5dede]")
                loading_screen.update_progress(80)
                await self.force_ui_update()

                loading_screen.update_loading_log("[#f5dede]Checking existing index...[/#f5dede]")
                await self.force_ui_update()

                # Build cache asynchronously to prevent UI freezing
                cache_result, cache_status = await asyncio.to_thread(
                    load_or_build_cache, model, current_note_paths, args.rebuild_cache
                )
                cache = cache_result
                loading_screen.update_progress(95)

                loading_screen.update_loading_log(
                    f"[#90ee90]✓ Index ready with {len(cache)} documents ({cache_status})[/#90ee90]", True
                )
                await self.force_ui_update()
            else:
                # Test mode - use dummy data
                loading_screen.update_status("Loading test data...")
                loading_screen.update_loading_log("[#f5dede]Loading dummy data...[/#f5dede]")
                loading_screen.update_progress(25)
                await self.force_ui_update()

                model = None
                cache = self.create_test_cache()
                loading_screen.update_progress(75)

                loading_screen.update_loading_log("[#90ee90]✓ Test data loaded[/#90ee90]", True)
                await self.force_ui_update()

            # Step 4: Initialize interface
            loading_screen.update_status("Starting application...")
            loading_screen.update_loading_log("[#f5dede]Initializing interface...[/#f5dede]")
            loading_screen.update_progress(100)
            await self.force_ui_update()

            loading_screen.update_loading_log("[bold #90ee90]✓ Ready to search![/bold #90ee90]")
            # In test mode, switch immediately; otherwise use timer
            if self.test_mode:
                self.show_main_interface()
            else:
                self.set_timer(0.5, self.show_main_interface)

        except Exception as e:
            loading_screen.update_progress(0)
            loading_screen.update_loading_log("[bold #ff6b6b]Error during initialization[/bold #ff6b6b]", True)
            loading_screen.update_loading_log("")
            loading_screen.update_loading_log(f"[#ff6b6b]{str(e)}[/#ff6b6b]")
            loading_screen.update_loading_log("")
            loading_screen.update_loading_log("[dim]Press ctrl+q to quit[/dim]")

    async def force_ui_update(self) -> None:
        """Force UI update to show loading progress immediately."""
        # Use a very short timer to allow UI to refresh
        await asyncio.sleep(0.01)

    def create_test_cache(self):
        """Create dummy cache data for testing."""
        import torch
        dummy_cache = {}

        # Create some sample notes
        test_notes = [
            ("test_data/test1.md", "# Test Note 1\n\nThis is a test note about machine learning and artificial intelligence. It covers topics like neural networks, deep learning, and natural language processing."),
            ("test_data/test2.md", "# Test Note 2\n\nThis note discusses Python programming, data structures, algorithms, and software development practices. It includes examples of object-oriented programming."),
            ("test_data/test3.md", "# Test Note 3\n\nA comprehensive guide to web development including HTML, CSS, JavaScript, and modern frameworks like React and Vue.js."),
        ]

        for note_path, content in test_notes:
            # Create dummy embedding (random tensor)
            dummy_embedding = torch.randn(768)  # BGE-base-en-v1.5 has 768 dimensions
            dummy_cache[note_path] = (content, dummy_embedding)

        return dummy_cache

    def update_loading_log(self, log, message, clear_first=False):
        if clear_first:
            log.clear()
        log.write(message)

    def show_main_interface(self) -> None:
        self.loading = False

        # Switch from loading screen to search screen
        self.switch_screen("search")

        # Initialize the search screen immediately
        self.initialize_search_screen()

    def initialize_search_screen(self) -> None:
        """Initialize the search screen after it has been composed."""
        # Get the search screen and initialize it
        search_screen = cast(SearchScreen, self.screen)
        results_log = search_screen.query_one("#results", RichLog)
        results_log.clear()

        if cache and len(cache) > 0:
            results_log.write(
                f"[bold #cd5c5c]Ready![/bold #cd5c5c] {len(cache)} notes indexed"
            )
            results_log.write("")
            results_log.write(
                f"[dim]Showing up to {MAX_RESULTS} results with score ≥ {SCORE_THRESHOLD}[/dim]"
            )
            results_log.write("[dim]Use ↑↓ arrows to navigate results[/dim]")
            results_log.write("[dim]Press ctrl+q to quit[/dim]")
        else:
            results_log.write("[bold #ff6b6b]No notes found![/bold #ff6b6b]")
            results_log.write("")
            results_log.write(
                "[dim]Make sure you have notes in your notes directory[/dim]"
            )



    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in modal screens."""
        if self.loading:
            return

        if event.button.id == "confirm-btn":
            # This is the confirm button in the analyze confirmation dialog
            pass  # Handled by ConfirmAnalyzeScreen
        elif event.button.id == "cancel-btn":
            # This is the cancel button in the analyze confirmation dialog
            pass  # Handled by ConfirmAnalyzeScreen

    def perform_search(self, query: str) -> None:
        global model, cache

        if not cache or self.loading:
            return

        # In test mode, model is None, so we need to handle search differently
        if not model:
            # Test mode: return dummy results based on query matching
            results = self.perform_test_search(query)
        else:
            results = search(query, model, cache, max_results=MAX_RESULTS)

        self.current_results = results
        self.selected_index = 0

        self.display_results()

        if results:
            self.display_preview(0)

    def perform_test_search(self, query: str) -> list:
        """Perform search in test mode without real model."""
        global cache

        if not query.strip() or not cache:
            return []

        results = []
        query_lower = query.lower()

        for path, (content, embedding) in cache.items():
            content_lower = content.lower()
            # Simple text matching for test mode
            if query_lower in content_lower:
                # Create a dummy score based on how well it matches
                score = 0.8 if query_lower in content_lower else 0.5
                results.append((score, path, content))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:MAX_RESULTS]

    def clear_results(self) -> None:
        self.current_results = []
        self.selected_index = 0

        try:
            results_log = self.query_one("#results", RichLog)
            preview_log = self.query_one("#preview", RichLog)
            results_log.clear()
            preview_log.clear()
        except:
            pass

    def display_results(self) -> None:
        try:
            results_log = self.query_one("#results", RichLog)
            results_log.clear()

            if not self.current_results:
                if self.app_mode == MODE_ANALYZE:
                    results_log.write("No wikilink suggestions found")
                else:
                    results_log.write("No results found")
                return

            search_input = self.query_one("#search-input", Input)
            query = search_input.value if search_input.value else ""

            if self.app_mode == MODE_ANALYZE:
                self.write_analysis_info(results_log, query)
                self.write_wikilink_suggestions(results_log)
            else:
                high_relevance = [
                    r for r in self.current_results if r[0] >= SCORE_THRESHOLD
                ]
                self.write_search_info(results_log, query, high_relevance)
                self.write_results_list(results_log)

        except Exception as e:
            print(f"Error displaying results: {e}")
            pass

    def write_search_info(self, log, query, high_relevance):
        log.write(f"[dim]Searching {get_notes_dir()} on {MODEL_NAME}[/dim]")
        log.write("")

        if query:
            log.write(
                f'[bold #cd5c5c]Found {len(high_relevance)} results for "{query}"[/bold #cd5c5c]'
            )
        else:
            log.write(
                f"[bold #cd5c5c]Found {len(high_relevance)} results[/bold #cd5c5c]"
            )
        log.write("")

    def write_results_list(self, log):
        for i, (score, path, content) in enumerate(self.current_results):
            rel_path = str(path).replace(str(Path.home()), "~")
            preview = content.replace("\n", " ").strip()[:100]
            if len(content) > 100:
                preview += "..."

            is_low_relevance = score < SCORE_THRESHOLD

            if i == self.selected_index:
                self.write_selected_result(
                    log, score, rel_path, preview, is_low_relevance
                )
            else:
                self.write_unselected_result(
                    log, score, rel_path, preview, is_low_relevance
                )

    def write_analysis_info(self, log, query):
        log.write(f"[dim]Analyzing text for wikilink suggestions[/dim]")
        log.write("")
        
        if query:
            log.write(f'[bold #cd5c5c]Found {len(self.current_results)} wikilink suggestions for "{query}"[/bold #cd5c5c]')
        else:
            log.write(f"[bold #cd5c5c]Found {len(self.current_results)} wikilink suggestions[/bold #cd5c5c]")
        log.write("")
        
    def write_wikilink_suggestions(self, log):
        """Display wikilink suggestions in the results panel."""
        if hasattr(self, 'current_suggestions') and self.current_suggestions:
            for i, suggestion in enumerate(self.current_suggestions):
                score = suggestion['score']
                candidate = suggestion['candidate']
                filename = suggestion['filename']
                wikilink = suggestion['wikilink']
                
                is_selected = i == self.selected_index
                
                if is_selected:
                    log.write(f"[bold #f5dede on #5d2828]{score:.3f}  {wikilink}[/]")
                    log.write(f"[#f5dede on #5d2828]   Candidate: {candidate}[/]")
                else:
                    log.write(f"[#cd5c5c]{score:.3f}  {wikilink}")
                    log.write(f"[dim]   Candidate: {candidate}[/dim]")
                
                log.write("")
        else:
            # Fallback if no suggestions data
            for i, (score, filename, wikilink) in enumerate(self.current_results):
                is_selected = i == self.selected_index
                
                if is_selected:
                    log.write(f"[bold #f5dede on #5d2828]{score:.3f}  {wikilink}[/]")
                else:
                    log.write(f"[#cd5c5c]{score:.3f}  {wikilink}")
                
                log.write("")

    def write_selected_result(self, log, score, path, preview, is_low_relevance):
        if is_low_relevance:
            log.write(f"[bold #f5dede on #5d2828][dim]{score:.3f}[/dim]  {path}[/]")
        else:
            log.write(f"[bold #f5dede on #5d2828]{score:.3f}  {path}[/]")
        log.write(f"[#f5dede on #5d2828]   {preview}[/]")
        log.write("")

    def write_unselected_result(self, log, score, path, preview, is_low_relevance):
        if is_low_relevance:
            log.write(f"[dim]{score:.3f}  {path}[/dim]")
        else:
            log.write(f"[#cd5c5c]{score:.3f}  {path}")
        log.write(f"[dim]   {preview}[/dim]")
        log.write("")

    def display_preview(self, index: int) -> None:
        try:
            preview_area = self.query_one("#preview", TextArea)

            if not (0 <= index < len(self.current_results)):
                preview_area.clear()
                return

            if self.app_mode == MODE_ANALYZE:
                analyze_screen = cast(AnalyzeScreen, self.screen)
                analyze_screen.display_analysis_preview(index)
            else:
                self.display_search_preview(index, preview_area)

        except:
            pass

    def display_analysis_results(self) -> None:
        """Display wikilink suggestions for the current note."""
        try:
            results_log = self.query_one("#results", RichLog)
            results_log.clear()

            if not self.current_analysis_results:
                results_log.write("No wikilink suggestions found for this note")
                return

            # Get the current note info
            if self.current_results and 0 <= self.selected_index < len(self.current_results):
                score, note_path, note_content = self.current_results[self.selected_index]
                filename = Path(note_path).name
                results_log.write(f"[bold #cd5c5c]Wikilink suggestions for: {filename}[/bold #cd5c5c]")
            else:
                results_log.write("[bold #cd5c5c]Wikilink suggestions[/bold #cd5c5c]")
            
            results_log.write("")
            results_log.write(f"Found {len(self.current_analysis_results)} suggestions")
            results_log.write("")

            # Display wikilink suggestions
            for i, (score, filename, wikilink) in enumerate(self.current_analysis_results):
                is_selected = i == self.selected_suggestion_index
                
                if is_selected:
                    results_log.write(f"[bold #f5dede on #5d2828]{score:.3f}  {wikilink}[/]")
                else:
                    results_log.write(f"[#cd5c5c]{score:.3f}  {wikilink}")
                
                results_log.write("")

        except Exception as e:
            print(f"Error displaying analysis results: {e}")

    def display_search_preview(self, index: int, preview_area: TextArea) -> None:
        """Display preview for search mode."""
        score, path, content = self.current_results[index]
        rel_path = str(path).replace(str(Path.home()), "~")

        # Combine header and content
        header = f"{rel_path}\n"
        header += f"Score: {score:.4f}\n\n"
        full_content = header + content

        preview_area.clear()
        preview_area.load_text(full_content)
        preview_area.scroll_home()

    def on_key(self, event) -> None:
        if self.loading:
            return

        if event.key in ("up", "down"):
            event.prevent_default()
            event.stop()

            if self.app_mode == MODE_SEARCH:
                # Search mode navigation
                if not self.current_results:
                    return
                
                if event.key == "up":
                    self.selected_index = max(0, self.selected_index - 1)
                else:
                    self.selected_index = min(
                        len(self.current_results) - 1, self.selected_index + 1
                    )

                self.display_results()
                self.display_preview(self.selected_index)
                
            elif self.app_mode == MODE_ANALYZE:
                # Analyze mode navigation through ALL suggestions
                if not self.all_analysis_suggestions:
                    return
                
                if event.key == "up":
                    self.selected_suggestion_index = max(0, self.selected_suggestion_index - 1)
                else:
                    self.selected_suggestion_index = min(
                        len(self.all_analysis_suggestions) - 1, self.selected_suggestion_index + 1
                    )

                analyze_screen = cast(AnalyzeScreen, self.screen)
                analyze_screen.display_all_analysis_results()
                analyze_screen.display_analysis_preview(self.selected_suggestion_index)

    def debounce_search(self, query):
        if hasattr(self, "_search_timer") and self._search_timer:
            self._search_timer.stop()

        self._search_timer = self.set_timer(0.3, lambda: self.perform_search(query))

    def on_rich_log_click(self, event) -> None:
        if not self.current_results or self.loading:
            return

        if event.widget.id != "results":
            return

        clicked_line = event.y

        # Skip header lines (4 lines: directory, empty, result count, empty)
        if clicked_line < 4:
            return

        # Each result takes 3 lines (score+path, preview, empty line)
        result_line = clicked_line - 4
        result_index = result_line // 3

        if 0 <= result_index < len(self.current_results):
            self.selected_index = result_index
            self.display_results()
            self.display_preview(self.selected_index)
            
            # If in analyze mode, automatically analyze the selected note
            if self.app_mode == MODE_ANALYZE:
                self.analyze_current_note()
            
            # If in analyze mode, automatically analyze the selected note
            if self.app_mode == MODE_ANALYZE:
                self.analyze_current_note()
            
            # If in analyze mode, automatically analyze the selected note
            if self.app_mode == MODE_ANALYZE:
                self.analyze_current_note()
            
            # If in analyze mode, automatically analyze the selected note
            if self.app_mode == MODE_ANALYZE:
                self.analyze_current_note()


    def toggle_app_mode(self) -> None:
        """Switch between search and analyze modes."""
        if self.app_mode == MODE_SEARCH:
            # Show confirmation dialog before switching to analyze mode (unless in test mode)
            if not self.test_mode:
                global cache
                note_count = len(cache) if cache else 0

                def on_confirm(confirmed: bool | None) -> None:
                    if confirmed:
                        self.app_mode = MODE_ANALYZE
                        self.switch_screen("analyze")
                        # Start the scan asynchronously
                        self.call_later(self.scan_all_notes_for_wikilinks)
                    # Clear current results when switching modes
                    self.clear_results()

                self.push_screen(ConfirmAnalyzeScreen(note_count), on_confirm)
                return

            self.app_mode = MODE_ANALYZE
            self.switch_screen("analyze")
            # Start the scan asynchronously
            self.call_later(self.scan_all_notes_for_wikilinks)
        else:
            self.app_mode = MODE_SEARCH
            self.switch_screen("search")

        # Clear current results when switching modes
        self.clear_results()



    async def scan_all_notes_for_wikilinks(self) -> None:
        """Scan ALL notes in the folder and generate wikilink suggestions with progress."""
        global model, cache

        if not model or not cache or self.loading:
            return

        print("Scanning all notes for wikilink suggestions...")

        try:
            # Get the analyze screen
            analyze_screen = cast(AnalyzeScreen, self.screen)

            # Initialize progress
            total_notes = len(cache)
            analyze_screen.update_progress(0, f"Preparing to scan {total_notes} notes...")

            all_suggestions = []
            processed_count = 0

            # Process each note with progress updates
            for note_path, (note_content, note_embedding) in cache.items():
                filename = Path(note_path).name

                # Update progress
                progress_percent = int((processed_count / total_notes) * 100)
                analyze_screen.update_progress(progress_percent, f"Analyzing: {filename}")

                # Load spaCy model if available
                nlp = load_spacy_model()

                # Analyze this note for wikilink candidates
                suggestions = analyze_text_for_wikilinks(note_content, model, cache, nlp)

                # Add note context to each suggestion
                for suggestion in suggestions:
                    suggestion['source_note'] = filename
                    suggestion['source_note_path'] = note_path
                    suggestion['source_note_content'] = note_content[:200] + "..." if len(note_content) > 200 else note_content

                all_suggestions.extend(suggestions)
                processed_count += 1

                # Small delay to allow UI updates
                await asyncio.sleep(0.01)

            # Sort by score
            all_suggestions.sort(key=lambda x: x['score'], reverse=True)

            self.all_analysis_suggestions = all_suggestions
            self.selected_suggestion_index = 0

            analyze_screen.update_progress(100, f"Scan complete! Found {len(all_suggestions)} suggestions")
            await asyncio.sleep(1.0)  # Show completion message briefly

            analyze_screen = cast(AnalyzeScreen, self.screen)
            analyze_screen.display_all_analysis_results()

            if all_suggestions:
                analyze_screen.display_analysis_preview(0)

            print(f"Found {len(all_suggestions)} total wikilink suggestions")

        except Exception as e:
            analyze_screen = cast(AnalyzeScreen, self.screen)
            analyze_screen.update_progress(0, f"Error during scan: {str(e)}")
            print(f"Error scanning all notes: {e}")
            self.all_analysis_suggestions = []
            analyze_screen.display_all_analysis_results()




    def analyze_current_note(self) -> None:
        """Analyze the currently viewed note for wikilink candidates."""
        global model, cache

        if not model or not cache or self.loading or not self.current_results:
            return

        try:
            # Switch to search screen for displaying analysis results
            self.switch_screen("search")
            self.app_mode = MODE_SEARCH  # Reset mode since we're switching screens

            # Get the currently selected note content
            current_index = self.selected_index
            if current_index >= len(self.current_results):
                return

            score, note_path, note_content = self.current_results[current_index]

            # Load spaCy model if available
            nlp = load_spacy_model()

            # Analyze the note content for wikilink candidates
            suggestions = analyze_text_for_wikilinks(note_content, model, cache, nlp)

            # Convert suggestions to the format expected by display methods
            self.current_analysis_results = [(s['score'], s['filename'], s['wikilink']) for s in suggestions]
            self.current_suggestions = suggestions  # Store full suggestions for preview
            self.selected_suggestion_index = 0

            self.display_analysis_results()

            if suggestions:
                self.display_preview(self.selected_index)

        except Exception as e:
            print(f"Error analyzing current note: {e}")
            self.current_analysis_results = []
            self.current_suggestions = []
            # Switch to search screen for error display
            self.switch_screen("search")
            self.app_mode = MODE_SEARCH
            self.display_analysis_results()

    def analyze_text(self, text: str) -> None:
        """Handle text input based on current mode."""
        if self.app_mode == MODE_SEARCH:
            self.perform_search(text)
        elif self.app_mode == MODE_ANALYZE:
            # In analyze mode, user input should still work for search
            # But analysis happens automatically on note selection
            self.perform_search(text)

    def debounce_analysis(self, query):
        """Debounced text analysis for real-time wikilink suggestions."""
        if hasattr(self, "_analysis_timer") and self._analysis_timer:
            self._analysis_timer.stop()

        self._analysis_timer = self.set_timer(0.3, lambda: self.analyze_text(query))


def main():
    args = parse_arguments()
    notes_dir = Path(args.notes_dir)

    print(f"Using notes directory: {notes_dir}")

    if not notes_dir.is_dir():
        print(f"Error: Notes directory not found at '{notes_dir}'")
        print(
            f"Create the directory or specify a different one: python main.py /path/to/notes"
        )
        sys.exit(1)

    app = SearchApp()
    app.test_mode = args.test_mode  # Set test mode from arguments
    app.run()


if __name__ == "__main__":
    main()
