"""UI components for the semantic note search application."""

import asyncio
from pathlib import Path
from typing import cast

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, RichLog, TextArea, Button, ProgressBar, Label
from textual.binding import Binding
from textual.screen import Screen, ModalScreen

from config import MODE_SEARCH, MODE_ANALYZE
from ai import model, cache, load_model, get_all_notes, load_or_build_cache, analyze_text_for_wikilinks, search, load_spacy_model


class CustomInput(Input):
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all"),
        Binding("ctrl+backspace", "delete_left_word", "Delete previous word"),
        Binding("ctrl+delete", "delete_right_word", "Delete next word"),
    ]
    cursor_blink = True


class LoadingScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="loading-screen"):
            with Vertical(id="loading-content"):
                yield Label("🔍 Semantic Note Search", id="loading-title")
                yield Label("", id="loading-subtitle")
                with Vertical(id="loading-progress-section"):
                    yield Label("Loading components...", id="loading-status")
                    yield ProgressBar(id="loading-progress", total=100)



    def update_status(self, status: str):
        self.query_one("#loading-status", Label).update(status)

    def update_subtitle(self, subtitle: str):
        self.query_one("#loading-subtitle", Label).update(subtitle)

    def update_progress(self, progress: int):
        self.query_one("#loading-progress", ProgressBar).progress = progress

    def action_quit(self) -> None:
        self.app.exit()


class SearchScreen(Screen):

    def compose(self) -> ComposeResult:
        with Vertical(classes="main-container"):
            with Horizontal(classes="top-bar"):
                yield Label("🔍 Semantic Search", id="mode-label")
                yield Button("Analyze", id="mode-btn", classes="mode-button")
            
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

    def on_mount(self):
        self.focus_input()

    def focus_input(self):
        try:
            self.query_one("#search-input", Input).focus()
        except:
            pass

    async def on_input_submitted(self, event) -> None:
        app = cast(SearchApp, self.app)
        if app.loading:
            return

        query = event.value.strip()
        if query:
            if app.app_mode == MODE_SEARCH:
                await app.perform_search(query)
            else:
                await app.analyze_text(query)
        else:
            app.clear_results()

        self.focus_input()

    async def on_input_changed(self, event) -> None:
        app = cast(SearchApp, self.app)
        if app.loading:
            return

        query = event.value.strip()
        if query and len(query) >= 1:
            if app.app_mode == MODE_SEARCH:
                if app.test_mode:
                    await app.perform_search(query)
                else:
                    app.debounce_search(query)
            else:
                if app.test_mode:
                    await app.analyze_text(query)
                else:
                    app.debounce_analysis(query)
        elif not query:
            app.clear_results()

    def on_button_pressed(self, event) -> None:
        app = cast(SearchApp, self.app)
        if app.loading:
            return

        if event.button.id == "mode-btn":
            app.call_later(app.toggle_app_mode)


class AnalyzeScreen(Screen):

    def compose(self) -> ComposeResult:
        with Vertical(classes="main-container"):
            with Horizontal(classes="top-bar"):
                yield Label("📝 Analyze Mode", id="mode-label")
                yield Button("Search", id="mode-btn", classes="mode-button")
            
            with Vertical(classes="analyze-container", id="analyze-layout"):
                with Vertical(id="analyze-progress-section"):
                    yield Label("Scan Progress", id="progress-label")
                    yield ProgressBar(id="analyze-progress", total=100)
                    yield Label("Ready to scan...", id="progress-status")

                with Horizontal(classes="analyze-main-content"):
                    with Vertical(classes="analyze-suggestions-panel"):
                        yield RichLog(id="analyze-results", auto_scroll=False, markup=True)
                    with Vertical(classes="analyze-preview-container"):
                        with Vertical(classes="analyze-source-panel"):
                            yield TextArea(
                                id="analyze-source",
                                read_only=True,
                                language="markdown",
                                theme="dracula",
                            )
                        with Vertical(classes="analyze-target-panel"):
                            yield TextArea(
                                id="analyze-target",
                                read_only=True,
                                language="markdown",
                                theme="dracula",
                            )

    def on_button_pressed(self, event) -> None:
        app = cast(SearchApp, self.app)
        if app.loading:
            return

        if event.button.id == "mode-btn":
            app.call_later(app.toggle_app_mode)

    def update_progress(self, percentage: int, status: str):
        try:
            self.query_one("#analyze-progress", ProgressBar).progress = percentage
            self.query_one("#progress-status", Label).update(status)
        except Exception as e:
            print(f"Error updating progress: {e}")

    def display_all_analysis_results(self) -> None:
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

            for i, suggestion in enumerate(app.all_analysis_suggestions):
                score = suggestion['score']
                source_note = suggestion['source_note']
                candidate = suggestion['candidate']
                target_title = self.get_note_title(suggestion['note_content'], suggestion['filename'])
                is_selected = i == app.selected_suggestion_index

                if is_selected:
                    results_log.write(f"[bold #f5dede on #5d2828]{score:.3f}  {source_note} ({candidate})[/]")
                    results_log.write(f"[#f5dede on #5d2828]   -> {suggestion['filename']} ({target_title})[/]")
                else:
                    results_log.write(f"[#cd5c5c]{score:.3f}  {source_note} ({candidate})")
                    results_log.write(f"[dim]   -> {suggestion['filename']} ({target_title})[/dim]")

        except Exception as e:
            print(f"Error displaying all analysis results: {e}")

    def display_analysis_preview(self, index: int) -> None:
        try:
            app = cast(SearchApp, self.app)
            source_area = self.query_one("#analyze-source", TextArea)
            target_area = self.query_one("#analyze-target", TextArea)
            source_area.clear()
            target_area.clear()

            if not (0 <= index < len(app.all_analysis_suggestions)):
                source_area.load_text("No suggestion selected")
                target_area.load_text("No suggestion selected")
                return

            suggestion = app.all_analysis_suggestions[index]
            source_note_path = suggestion['source_note_path']
            
            global cache
            if cache and source_note_path in cache:
                full_source_content = cache[source_note_path][0]
            else:
                full_source_content = suggestion['source_note_content']

            wikilink = suggestion['wikilink']
            candidate = suggestion['candidate']

            # Parse wikilink to get display text
            import re
            wikilink_match = re.match(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', wikilink)
            if wikilink_match:
                link_target = wikilink_match.group(1)
                display_text = wikilink_match.group(2) if wikilink_match.group(2) else link_target
            else:
                display_text = candidate

            pattern = re.compile(re.escape(wikilink), re.IGNORECASE)
            match = pattern.search(full_source_content)
            if match:
                highlighted_wikilink = f"[bold #f5dede on #8b3a3a]{display_text}[/bold #f5dede on #8b3a3a]"
                highlighted_content = full_source_content[:match.start()] + highlighted_wikilink + full_source_content[match.end():]
                lines_before = full_source_content[:match.start()].count('\n')
            else:
                highlighted_content = full_source_content
                lines_before = 0

            source_content = highlighted_content

            source_area.load_text(source_content)

            linked_filename = suggestion['filename']
            target_content = ""

            if cache:
                linked_note_path = None
                for path in cache.keys():
                    if Path(path).name == linked_filename:
                        linked_note_path = path
                        break

                if linked_note_path and linked_note_path in cache:
                    linked_content = cache[linked_note_path][0]
                    target_content += linked_content
                else:
                    target_content += f"Content not found for {linked_filename}"
            else:
                target_content += "Cache not available"

            target_area.load_text(target_content)

            if match:
                source_area.move_cursor((lines_before + 5, 0))
                source_area.scroll_cursor_visible()
            else:
                source_area.scroll_home()
            
            target_area.scroll_home()

        except Exception as e:
            pass


class ConfirmAnalyzeScreen(ModalScreen[bool]):
    def __init__(self, note_count: int):
        super().__init__()
        self.note_count = note_count

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(f"Warning: Analyze Mode Scan Confirmation", id="confirm-title")
            yield Label("")
            yield Label(f"This will scan {self.note_count} notes for wikilink suggestions.", id="confirm-message")
            yield Label("This process may take several minutes depending on the number of notes.", id="confirm-warning")
            yield Label("")
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Start Scan", id="confirm-btn", variant="primary")

    def on_button_pressed(self, event) -> None:
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        elif event.button.id == "cancel-btn":
            self.dismiss(False)

    CSS = """
    #confirm-dialog {
        width: 80;
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

    CSS = """
    Screen { background: #1a0d0d; }
    .main-container { height: 100%; }
    .top-bar {
        height: 5;
        background: #2d1414;
        border: solid #8b3a3a;
        margin: 1 1 0 1;
        padding: 0 1;
        align: center middle;
    }
    #mode-label {
        color: #f5dede;
        text-style: bold;
        width: 1fr;
    }
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
        margin: 0 1 0 1;
        background: #2d1414;
    }
    .analyze-preview-container {
        width: 1fr;
        height: 1fr;
        margin: 0 1 0 0;
        background: #2d1414;
    }
    .analyze-source-panel {
        height: 1fr;
        margin: 0 0 1 0;
        background: #2d1414;
    }
    .analyze-target-panel {
        height: 1fr;
        margin: 1 0 0 0;
        background: #2d1414;
    }
    #source-label, #target-label {
        text-align: center;
        color: $primary;
        text-style: bold;
        margin: 0 0 1 0;
    }
    #analyze-source, #analyze-target {
        height: 1fr;
        scrollbar-size: 1 1;
        color: #f5dede;
        background: #2d1414;
        overflow-y: auto;
        border: solid #8b3a3a;
    }
    .search-container {
        height: auto;
        background: #2d1414;
        padding: 0 1;
        margin: 0 1 1 1;
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
        width: 10;
        min-width: 10;
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
    #results, #analyze-results {
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
        width: auto;
        height: auto;
        background: transparent;
        padding: 2;
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
        self.all_analysis_suggestions = []
        self.test_mode = False
        self._notes_dir: Path | None = None

    def compose(self) -> ComposeResult:
        return
        yield

    def on_mount(self) -> None:
        self.push_screen("loading")
        self.set_timer(0.05, self.start_async_initialization)

    def start_async_initialization(self) -> None:
        self.call_later(self.initialize_app_async)

    async def initialize_app_async(self) -> None:
        global model, cache

        loading_screen = cast(LoadingScreen, self.screen)
        loading_screen.update_progress(0)

        try:
            args = self.parse_arguments()

            loading_screen.update_status("Initializing...")
            await self.force_ui_update()

            if not self.test_mode:
                loading_screen.update_status("Checking GPU availability...")
                loading_screen.update_progress(5)
                await self.force_ui_update()
                
                from ai import get_gpu_status
                gpu_status = get_gpu_status()
                
                if gpu_status['available']:
                    subtitle = f"🚀 Using GPU: {gpu_status['device_name']}"
                else:
                    subtitle = "💻 Using CPU (install CUDA for GPU acceleration)"
                
                loading_screen.update_subtitle(subtitle)
                loading_screen.update_progress(10)
                await self.force_ui_update()

                loading_screen.update_status("Loading AI model...")
                loading_screen.update_progress(15)
                await self.force_ui_update()

                model = load_model()
                if model is None:
                    raise Exception("Failed to load AI model. Please check your installation and try again.")
                loading_screen.update_progress(40)
                await self.force_ui_update()

                loading_screen.update_status("Scanning notes...")
                loading_screen.update_progress(50)
                await self.force_ui_update()

                current_notes = get_all_notes(self.get_notes_dir())
                current_note_paths = {str(p) for p in current_notes}
                loading_screen.update_status(f"Found {len(current_notes)} notes...")
                loading_screen.update_progress(70)
                await self.force_ui_update()

                loading_screen.update_status("Indexing notes...")
                loading_screen.update_progress(80)
                await self.force_ui_update()

                def progress_callback(progress, status):
                    loading_screen.update_progress(progress)
                    loading_screen.update_status(status)

                import ai
                ai_cache, cache_status = await load_or_build_cache(
                    model, current_note_paths, self.get_notes_dir(), args.rebuild_cache, progress_callback=progress_callback
                )
                cache = ai_cache
                loading_screen.update_progress(95)
            else:
                loading_screen.update_status("Loading test data...")
                loading_screen.update_progress(25)
                await self.force_ui_update()

                model = None
                import ai
                ai.cache = self.create_test_cache()
                loading_screen.update_progress(75)
                await self.force_ui_update()

            loading_screen.update_status("Starting application...")
            loading_screen.update_progress(100)
            await self.force_ui_update()
            self.show_main_interface()

        except Exception as e:
            loading_screen.update_progress(0)
            loading_screen.update_status(f"Error: {str(e)}")

    async def force_ui_update(self) -> None:
        await asyncio.sleep(0.01)

    def create_test_cache(self):
        import torch
        from ai import get_all_notes
        
        notes_dir = self.get_notes_dir()
        actual_notes = get_all_notes(notes_dir)
        dummy_cache = {}

        for note_path in actual_notes:
            if note_path.name == "test1.md":
                content = "# Test Note 1\n\nThis is a test note about machine learning and artificial intelligence. It covers topics like neural networks, deep learning, and natural language processing."
            elif note_path.name == "ml_notes.md":
                content = "# Machine Learning Notes\n\nThis note covers machine learning concepts including supervised learning, unsupervised learning, neural networks, and deep learning architectures."
            elif note_path.name == "python_notes.md":
                content = "# Python Programming Notes\n\nThis note discusses Python programming, data structures, algorithms, and software development practices. It includes examples of object-oriented programming."
            else:
                content = f"# {note_path.stem}\n\nThis is a test note."
            
            dummy_embedding = torch.randn(768)
            dummy_cache[str(note_path)] = (content, dummy_embedding)

        return dummy_cache

    def show_main_interface(self) -> None:
        self.loading = False
        self.switch_screen("search")
        self.initialize_search_screen()
        self.update_mode_button()

    def initialize_search_screen(self) -> None:
        search_screen = cast(SearchScreen, self.screen)
        results_log = search_screen.query_one("#results", RichLog)
        results_log.clear()

        cache_size = len(cache) if cache else 0
        
        if cache and len(cache) > 0:
            results_log.write(
                f"[bold #cd5c5c]Ready![/bold #cd5c5c] {len(cache)} notes indexed"
            )
            results_log.write("")
            results_log.write(
                f"[dim]Showing up to {self.MAX_RESULTS} results with score ≥ {self.SCORE_THRESHOLD}[/dim]"
            )
            results_log.write("[dim]Use ↑↓ arrows to navigate results[/dim]")
            results_log.write("[dim]Press ctrl+q to quit[/dim]")
        else:
            results_log.write("[bold #ff6b6b]No notes found![/bold #ff6b6b]")
            results_log.write("")
            results_log.write(
                "[dim]Make sure you have notes in your notes directory[/dim]"
            )

    def on_button_pressed(self, event) -> None:
        if self.loading:
            return
        if event.button.id in ("confirm-btn", "cancel-btn"):
            pass
        elif event.button.id == "cancel-btn":
            # This is the cancel button in the analyze confirmation dialog
            pass  # Handled by ConfirmAnalyzeScreen

    async def perform_search(self, query: str) -> None:
        global model, cache

        if not cache or self.loading:
            return

        if self.test_mode:
            results = self.perform_test_search(query)
        else:
            results = await asyncio.to_thread(search, query, model, cache, max_results=self.MAX_RESULTS)

        self.current_results = results
        self.selected_index = 0

        self.display_results()
        self.screen.refresh()

        if results:
            self.display_preview(0)

    def perform_test_search(self, query: str) -> list:
        global cache

        if not query.strip() or not cache:
            return []

        results = []
        query_lower = query.lower()

        for path, (content, embedding) in cache.items():
            content_lower = content.lower()
            if query_lower in content_lower:
                score = 0.8
                results.append((score, path, content))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:self.MAX_RESULTS]

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
            results_id = "#results" if self.app_mode == MODE_SEARCH else "#analyze-results"
            results_log = self.screen.query_one(results_id, RichLog)
            results_log.clear()

            if not self.current_results:
                if self.app_mode == MODE_ANALYZE:
                    results_log.write("No wikilink suggestions found")
                else:
                    results_log.write("No results found")
                return

            search_input = self.screen.query_one("#search-input", Input)
            query = search_input.value if search_input.value else ""

            if self.app_mode == MODE_ANALYZE:
                self.write_analysis_info(results_log, query)
                self.write_wikilink_suggestions(results_log)
            else:
                high_relevance = [
                    r for r in self.current_results if r[0] >= self.SCORE_THRESHOLD
                ]
                self.write_search_info(results_log, query, high_relevance)
                self.write_results_list(results_log)

        except Exception as e:
            print(f"Error displaying results: {e}")
            pass

    def write_search_info(self, log, query, high_relevance):
        log.write(f"[dim]Searching {self.get_notes_dir()} on {self.MODEL_NAME}[/dim]")
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

            is_low_relevance = score < self.SCORE_THRESHOLD

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

    def get_note_title(self, content, filename):
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return Path(filename).stem

    def write_wikilink_suggestions(self, log):
        if hasattr(self, 'current_suggestions') and self.current_suggestions:
            for i, suggestion in enumerate(self.current_suggestions):
                score = suggestion['score']
                candidate = suggestion['candidate']
                filename = suggestion['filename']
                source_note = suggestion.get('source_note', 'Unknown')
                target_title = self.get_note_title(suggestion.get('note_content', ''), filename)
                is_selected = i == self.selected_index

                if is_selected:
                    log.write(f"[bold #f5dede on #5d2828]{score:.3f}  {source_note} ({candidate}) -> {filename} ({target_title})[/]")
                else:
                    log.write(f"[#cd5c5c]{score:.3f}  {source_note} ({candidate}) -> {filename} ({target_title})")

                log.write("")
        else:
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
            if self.app_mode == MODE_ANALYZE:
                analyze_screen = cast(AnalyzeScreen, self.screen)
                analyze_screen.display_analysis_preview(index)
            else:
                preview_area = self.screen.query_one("#preview", TextArea)

                if not (0 <= index < len(self.current_results)):
                    preview_area.clear()
                    return

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

        async def do_search():
            await self.perform_search(query)

        self._search_timer = self.set_timer(0.3, lambda: asyncio.create_task(do_search()))

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

    def toggle_app_mode(self) -> None:
        if self.app_mode == MODE_SEARCH:
            if not self.test_mode:
                global cache
                note_count = len(cache) if cache else 0

                def on_confirm(confirmed: bool | None) -> None:
                    if confirmed:
                        self.app_mode = MODE_ANALYZE
                        self.switch_screen("analyze")
                        self.call_later(self.scan_all_notes_for_wikilinks)
                        self.call_later(self.update_mode_button)

                self.push_screen(ConfirmAnalyzeScreen(note_count), on_confirm)
                return

            self.app_mode = MODE_ANALYZE
            self.switch_screen("analyze")
            self.call_later(self.scan_all_notes_for_wikilinks)
        else:
            self.app_mode = MODE_SEARCH
            self.switch_screen("search")
        
        self.call_later(self.update_mode_button)

    async def scan_all_notes_for_wikilinks(self) -> None:
        global model, cache

        if not model or not cache or self.loading:
            return

        try:
            analyze_screen = cast(AnalyzeScreen, self.screen)

            total_notes = len(cache)
            analyze_screen.update_progress(0, f"Preparing to scan {total_notes} notes...")

            all_suggestions = []
            processed_count = 0

            for note_path, (note_content, note_embedding) in cache.items():
                filename = Path(note_path).name

                progress_percent = int((processed_count / total_notes) * 100)
                analyze_screen.update_progress(progress_percent, f"Analyzing: {filename}")

                nlp = load_spacy_model()
                suggestions = analyze_text_for_wikilinks(note_content, model, cache, nlp, note_path)

                for suggestion in suggestions:
                    suggestion['source_note'] = filename
                    suggestion['source_note_path'] = note_path
                    suggestion['source_note_content'] = note_content[:200] + "..." if len(note_content) > 200 else note_content

                all_suggestions.extend(suggestions)
                processed_count += 1

                await asyncio.sleep(0.01)

            all_suggestions.sort(key=lambda x: x['score'], reverse=True)

            self.all_analysis_suggestions = all_suggestions
            self.selected_suggestion_index = 0

            analyze_screen.update_progress(100, f"Scan complete! Found {len(all_suggestions)} suggestions")
            await asyncio.sleep(1.0)

            analyze_screen = cast(AnalyzeScreen, self.screen)
            analyze_screen.display_all_analysis_results()

            if all_suggestions:
                analyze_screen.display_analysis_preview(0)

        except Exception as e:
            analyze_screen = cast(AnalyzeScreen, self.screen)
            analyze_screen.update_progress(0, f"Error during scan: {str(e)}")
            print(f"Error scanning all notes: {e}")
            self.all_analysis_suggestions = []
            analyze_screen.display_all_analysis_results()

    def analyze_current_note(self) -> None:
        global model, cache

        if not model or not cache or self.loading or not self.current_results:
            return

        try:
            self.switch_screen("search")
            self.app_mode = MODE_SEARCH

            current_index = self.selected_index
            if current_index >= len(self.current_results):
                return

            score, note_path, note_content = self.current_results[current_index]

            nlp = load_spacy_model()
            suggestions = analyze_text_for_wikilinks(note_content, model, cache, nlp, note_path)

            self.current_analysis_results = [(s['score'], s['filename'], s['wikilink']) for s in suggestions]
            self.current_suggestions = suggestions
            self.selected_suggestion_index = 0

            self.display_analysis_results()

            if suggestions:
                self.display_preview(self.selected_index)

        except Exception as e:
            print(f"Error analyzing current note: {e}")
            self.current_analysis_results = []
            self.current_suggestions = []
            self.switch_screen("search")
            self.app_mode = MODE_SEARCH
            self.display_analysis_results()

    async def analyze_text(self, text: str) -> None:
        """Handle text input based on current mode."""
        if self.app_mode == MODE_SEARCH:
            await self.perform_search(text)
        elif self.app_mode == MODE_ANALYZE:
            # In analyze mode, user input should still work for search
            # But analysis happens automatically on note selection
            await self.perform_search(text)

    def debounce_analysis(self, query):
        """Debounced text analysis for real-time wikilink suggestions."""
        if hasattr(self, "_analysis_timer") and self._analysis_timer:
            self._analysis_timer.stop()

        async def do_analysis():
            await self.analyze_text(query)

        self._analysis_timer = self.set_timer(0.3, lambda: asyncio.create_task(do_analysis()))

    # Properties for constants
    @property
    def MAX_RESULTS(self):
        from config import MAX_RESULTS
        return MAX_RESULTS

    @property
    def SCORE_THRESHOLD(self):
        from config import SCORE_THRESHOLD
        return SCORE_THRESHOLD

    @property
    def MODEL_NAME(self):
        from config import MODEL_NAME
        return MODEL_NAME

    def parse_arguments(self):
        import argparse
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

    def update_mode_button(self) -> None:
        """Update mode button text based on current mode."""
        try:
            mode_button = self.screen.query_one("#mode-btn", Button)
            if self.app_mode == MODE_SEARCH:
                mode_button.label = "Analyze"
            else:
                mode_button.label = "Search"
        except:
            pass

    @property
    def notes_dir(self) -> Path | None:
        return self._notes_dir

    @notes_dir.setter
    def notes_dir(self, value: Path) -> None:
        self._notes_dir = value

    def get_notes_dir(self):
        if self._notes_dir:
            return self._notes_dir
        # Fallback to parsing arguments if not set (for testing)
        args = self.parse_arguments()
        return Path(args.notes_dir)
