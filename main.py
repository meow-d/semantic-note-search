#!/usr/bin/env python3

import os
import pickle
import sys
import threading
from pathlib import Path

from textual.containers import Container

try:
    import torch
except ImportError:
    print("Error: 'torch' library not found.")
    print("Please install it by running: pip install torch")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("Error: 'sentence-transformers' library not found.")
    print("Please install it by running: pip install sentence-transformers torch")
    sys.exit(1)

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Input, RichLog, TextArea
    from textual.binding import Binding
except ImportError:
    print("Error: 'textual' library not found.")
    print("Please install it by running: pip install textual")
    sys.exit(1)

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Semantic note search using Sentence Transformers"
    )
    parser.add_argument(
        "notes_dir",
        nargs="?",
        default="notes",
        help="Directory containing note files (default: ./notes)",
    )
    return parser.parse_args()


def get_notes_dir():
    args = parse_arguments()
    return Path(args.notes_dir)


CACHE_FILE = Path(__file__).parent / "cache.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
NOTE_EXTENSIONS = {".txt", ".md"}
SCORE_THRESHOLD = 0.25
MAX_RESULTS = 12

model = None
cache = None


def load_model():
    global model
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
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
    print("Building cache... This may take a while for the first time.")
    cache = {}
    for note_path in notes:
        try:
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                embedding = model.encode(content, convert_to_tensor=True)
                cache[str(note_path)] = (content, embedding)
        except Exception as e:
            print(f"Warning: Could not read or process {note_path}: {e}")

    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cache built and saved to '{CACHE_FILE}'")
    except Exception as e:
        print(f"Error: Could not write cache file: {e}")

    return cache


def load_cache():
    print(f"Loading cache from {CACHE_FILE}")
    try:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return None


def update_cache_for_new_notes(model, cache, new_notes):
    for note_path in new_notes:
        try:
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                embedding = model.encode(content, convert_to_tensor=True)
                cache[note_path] = (content, embedding)
        except Exception as e:
            print(f"Warning: Could not process new note {note_path}: {e}")


def remove_deleted_notes_from_cache(cache, removed_notes):
    for path in removed_notes:
        cache.pop(path, None)


def load_or_build_cache(model, current_note_paths):
    cache = load_cache()

    if not cache:
        notes = [Path(p) for p in current_note_paths]
        return build_cache(notes, model)

    cached_note_paths = set(cache.keys())
    new_notes = current_note_paths - cached_note_paths
    removed_notes = cached_note_paths - current_note_paths

    if not new_notes and not removed_notes:
        print("Cache is up to date.")
    else:
        if new_notes:
            print(f"Adding {len(new_notes)} new notes to cache...")
            update_cache_for_new_notes(model, cache, new_notes)

        if removed_notes:
            print(f"Removing {len(removed_notes)} deleted notes from cache...")
            remove_deleted_notes_from_cache(cache, removed_notes)

        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)

    return cache


def search(
    query, model, cache, score_threshold=SCORE_THRESHOLD, max_results=MAX_RESULTS
):
    if not query.strip():
        return []

    print(f"Searching for: '{query}'")
    query_embedding = model.encode(query, convert_to_tensor=True)

    note_paths = list(cache.keys())
    note_embeddings_list = [data[1] for data in cache.values()]
    note_embeddings_tensor = torch.stack(note_embeddings_list)

    cosine_scores = util.cos_sim(query_embedding, note_embeddings_tensor)

    all_results = []
    for idx, score in enumerate(cosine_scores[0]):
        note_path = note_paths[idx]
        content = cache[note_path][0]
        all_results.append((score.item(), note_path, content))

    all_results.sort(key=lambda x: x[0], reverse=True)

    above_threshold = [r for r in all_results if r[0] >= score_threshold]

    if len(above_threshold) >= max_results:
        results = above_threshold
    else:
        results = all_results[:max_results]

    return results


class CustomInput(Input):
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all"),
        Binding("ctrl+backspace", "delete_left_word", "Delete previous word"),
        Binding("ctrl+delete", "delete_right_word", "Delete next word"),
    ]
    cursor_blink = True


class SearchApp(App):
    CSS = """
    Screen { background: #1a0d0d; }
    .main-container { height: 100%; }
    .content-container { height: 1fr; }
    .results-panel {
        width: 1fr;
        border: solid #8b3a3a;
        margin: 0 1 0 1;
        background: #2d1414;
    }
    .preview-panel {
        width: 1fr;
        border: solid #cd5c5c;
        margin: 0 1 0 0;
        background: #2d1414;
    }
    .search-container {
        height: auto;
        background: #2d1414;
        padding: 0 1;
    }
    #search-input {
        width: 100%;
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
        width: 60;
        height: auto;
        background: #2d1414;
        border: solid #8b3a3a;
        padding: 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.current_results = []
        self.selected_index = 0
        self.loading = True
        self._search_timer = None

    def compose(self) -> ComposeResult:
        with Container(id="loading-screen"):
            with Vertical(id="loading-content"):
                yield RichLog(id="loading-log", markup=True, auto_scroll=False)

        with Vertical(classes="main-container", id="main-interface"):
            with Horizontal(classes="content-container"):
                with Vertical(classes="results-panel"):
                    yield RichLog(id="results", auto_scroll=False, markup=True)
                with Vertical(classes="preview-panel"):
                    yield TextArea(
                        id="preview",
                        read_only=True,
                        language="markdown",
                        theme="dracula",
                    )

            with Container(classes="search-container"):
                yield CustomInput(placeholder="Enter search query...", id="search-input")

    def on_mount(self) -> None:
        self.set_timer(0.1, self.initialize_app)

    def initialize_app(self) -> None:
        global model, cache

        loading_log = self.query_one("#loading-log", RichLog)
        loading_log.clear()

        try:
            self.update_loading_log(
                loading_log, "[bold #cd5c5c]Semantic Note Search[/bold #cd5c5c]", True
            )
            self.update_loading_log(loading_log, "")

            loading_log.write(
                "[#f5dede]Step 1/4: Loading sentence transformer model...[/#f5dede]"
            )
            self.call_after_refresh(lambda: None)
            model = load_model()

            self.update_loading_log(
                loading_log, "[#90ee90]✓ Model loaded[/#90ee90]", True
            )
            self.update_loading_log(loading_log, "")

            loading_log.write("[#f5dede]Step 2/4: Scanning for notes...[/#f5dede]")
            self.call_after_refresh(lambda: None)
            current_notes = get_all_notes(get_notes_dir())
            current_note_paths = {str(p) for p in current_notes}

            self.update_loading_log(
                loading_log,
                f"[#90ee90]✓ Found {len(current_notes)} notes[/#90ee90]",
                True,
            )
            self.update_loading_log(loading_log, "")

            loading_log.write("[#f5dede]Step 3/4: Building search index...[/#f5dede]")
            self.call_after_refresh(lambda: None)
            cache = load_or_build_cache(model, current_note_paths)

            self.update_loading_log(
                loading_log, "[#90ee90]✓ Index ready[/#90ee90]", True
            )
            self.update_loading_log(loading_log, "")

            loading_log.write("[#f5dede]Step 4/4: Initializing interface...[/#f5dede]")
            self.call_after_refresh(lambda: None)

            loading_log.write("[bold #90ee90]✓ All set![/bold #90ee90]")
            self.set_timer(0.3, self.show_main_interface)

        except Exception as e:
            loading_log.clear()
            self.update_loading_log(
                loading_log,
                "[bold #ff6b6b]Error during initialization[/bold #ff6b6b]",
                True,
            )
            self.update_loading_log(loading_log, "")
            self.update_loading_log(loading_log, f"[#ff6b6b]{str(e)}[/#ff6b6b]")
            self.update_loading_log(loading_log, "")
            self.update_loading_log(loading_log, "[dim]Press ctrl+q to quit[/dim]")

    def update_loading_log(self, log, message, clear_first=False):
        if clear_first:
            log.clear()
        log.write(message)

    def show_main_interface(self) -> None:
        self.loading = False

        try:
            self.query_one("#loading-screen").display = False
            self.query_one("#main-interface").display = True
        except:
            pass

        results_log = self.query_one("#results", RichLog)
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

        self.focus_input()

    def focus_input(self):
        try:
            search_input = self.query_one("#search-input", Input)
            search_input.focus()
        except:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.loading:
            return

        query = event.value.strip()
        if query:
            self.perform_search(query)
        else:
            self.clear_results()

        self.focus_input()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.loading:
            return

        query = event.value.strip()
        if query and len(query) >= 2:
            self.debounce_search(query)
        elif not query:
            self.clear_results()

    def on_input_key(self, event) -> None:
        if self.loading:
            return
        self.focus_input()

    def perform_search(self, query: str) -> None:
        global model, cache

        if not model or not cache or self.loading:
            return

        results = search(query, model, cache)
        self.current_results = results
        self.selected_index = 0

        self.display_results()

        if results:
            self.display_preview(0)

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
                results_log.write("No results found")
                return

            search_input = self.query_one("#search-input", Input)
            query = search_input.value if search_input.value else ""
            high_relevance = [
                r for r in self.current_results if r[0] >= SCORE_THRESHOLD
            ]

            self.write_search_info(results_log, query, high_relevance)
            self.write_results_list(results_log)

        except:
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

            score, path, content = self.current_results[index]
            rel_path = str(path).replace(str(Path.home()), "~")

            # Combine header and content
            header = f"{rel_path}\n"
            header += f"Score: {score:.4f}\n\n"
            full_content = header + content

            preview_area.clear()
            preview_area.load_text(full_content)

            # Ensure scrolling stays at the top to prevent arrow key scroll issues
            preview_area.scroll_home()

        except:
            pass

    def on_key(self, event) -> None:
        if not self.current_results or self.loading:
            return

        if event.key in ("up", "down"):
            event.prevent_default()
            event.stop()

            if event.key == "up":
                self.selected_index = max(0, self.selected_index - 1)
            else:
                self.selected_index = min(
                    len(self.current_results) - 1, self.selected_index + 1
                )

            self.display_results()
            self.display_preview(self.selected_index)

    def debounce_search(self, query):
        if hasattr(self, "_search_timer") and self._search_timer:
            self._search_timer.cancel()

        self._search_timer = threading.Timer(0.3, lambda: self.perform_search(query))
        self._search_timer.start()

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


def main():
    args = parse_arguments()
    notes_dir = Path(args.notes_dir)

    print(f"Using notes directory: {notes_dir}")

    if not notes_dir.is_dir():
        print(f"Error: Notes directory not found at '{notes_dir}'")
        print(
            f"Create the directory or specify a different one: python {sys.argv[0]} /path/to/notes"
        )
        sys.exit(1)

    app = SearchApp()
    app.run()


if __name__ == "__main__":
    main()
