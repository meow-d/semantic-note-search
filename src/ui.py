"""UI components for the semantic note search application."""

import asyncio
from pathlib import Path
from typing import cast, List

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, RichLog, TextArea, Button, ProgressBar, Label
from textual.binding import Binding
from textual.screen import Screen, ModalScreen

from config import MODE_SEARCH, MODE_ANALYZE
from ai import (
    model,
    cache,
    load_model,
    get_all_notes,
    load_or_build_cache,
    analyze_text_for_wikilinks,
    search,
    load_spacy_model,
)


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
                yield CustomInput(
                    placeholder="Enter search query...", id="search-input"
                )

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
                app.debounce_search(query)
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
    BINDINGS = [
        Binding("ctrl+enter", "apply_suggestion", "Apply selected suggestion"),
    ]

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
                    yield Button(
                        "Apply Selected Suggestion",
                        id="apply-suggestion-btn",
                        disabled=True,
                    )

                with Horizontal(classes="analyze-main-content"):
                    with Vertical(classes="analyze-suggestions-panel"):
                        yield RichLog(
                            id="analyze-results", auto_scroll=False, markup=True
                        )
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
        elif event.button.id == "apply-suggestion-btn":
            self.apply_selected_suggestion()

    def apply_selected_suggestion(self) -> None:
        app = cast(SearchApp, self.app)
        if not app.all_analysis_suggestions or app.selected_suggestion_index >= len(
            app.all_analysis_suggestions
        ):
            return

        suggestion = app.all_analysis_suggestions[app.selected_suggestion_index]
        source_path = suggestion["source_note_path"]

        # Get the modified content from the source area
        source_area = self.query_one("#analyze-source", TextArea)
        modified_content = source_area.text

        # Write to file
        try:
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            # Update status
            self.query_one("#progress-status", Label).update(
                "Suggestion applied successfully!"
            )
        except Exception as e:
            self.query_one("#progress-status", Label).update(
                f"Error applying suggestion: {e}"
            )

    def action_apply_suggestion(self) -> None:
        self.apply_selected_suggestion()

    def update_progress(self, percentage: int, status: str):
        try:
            self.query_one("#analyze-progress", ProgressBar).progress = percentage
            self.query_one("#progress-status", Label).update(status)
        except Exception as e:
            print(f"Error updating progress: {e}")

    def get_note_title(self, content, filename):
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return Path(filename).stem

    def display_all_analysis_results(self) -> None:
        try:
            app = cast(SearchApp, self.app)
            results_log = self.query_one("#analyze-results", RichLog)
            results_log.clear()

            if not app.all_analysis_suggestions:
                results_log.write("No wikilink suggestions found")
                return

            results_log.write(
                f"[bold #cd5c5c]Wikilink Suggestions for All Notes[/bold #cd5c5c]"
            )
            results_log.write("")
            results_log.write(f"Found {len(app.all_analysis_suggestions)} suggestions")
            results_log.write("")
            results_log.write("[dim]Navigate with ↑↓ arrows[/dim]")
            results_log.write("")

            for i, suggestion in enumerate(app.all_analysis_suggestions):
                score = suggestion["score"]
                source_note = suggestion["source_note"]
                candidate = suggestion["candidate"]
                target_title = self.get_note_title(
                    suggestion["note_content"], suggestion["filename"]
                )
                is_selected = i == app.selected_suggestion_index

                if is_selected:
                    results_log.write(
                        f"[bold #f5dede on #5d2828]{score:.3f}  {source_note} ({candidate})[/]"
                    )
                    results_log.write(
                        f"[bold #f5dede on #5d2828]   -> {suggestion['filename']} ({target_title})[/]"
                    )
                else:
                    results_log.write(
                        f"[#cd5c5c]{score:.3f}  {source_note} ({candidate})"
                    )
                    results_log.write(
                        f"[#cd5c5c]   -> {suggestion['filename']} ({target_title})"
                    )
                results_log.write("")

            # Update apply button
            btn = self.query_one("#apply-suggestion-btn", Button)
            btn.disabled = len(app.all_analysis_suggestions) == 0

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
            source_note_path = suggestion["source_note_path"]

            global cache
            if cache and source_note_path in cache:
                full_source_content = cache[source_note_path][0]
            else:
                full_source_content = suggestion["source_note_content"]

            # Insert the suggested wikilink into the source content
            wikilink = suggestion["wikilink"]
            candidate = suggestion["candidate"]
            # Replace the first occurrence of the candidate with the wikilink syntax
            modified_source_content = full_source_content.replace(
                candidate, wikilink, 1
            )

            # For preview, show the raw [[ ]] syntax without processing existing wikilinks
            source_area.load_text(modified_source_content)

            # Highlight the inserted wikilink
            app.add_wikilink_highlights(
                source_area, modified_source_content, modified_source_content
            )

            # Find the position of the suggested wikilink in the content for cursor positioning
            import re

            wikilink_match = re.match(
                r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", suggestion["wikilink"]
            )
            match = None
            if wikilink_match:
                link_target = wikilink_match.group(1)
                display_text = (
                    wikilink_match.group(2) if wikilink_match.group(2) else link_target
                )
                display_pattern = re.compile(re.escape(display_text), re.IGNORECASE)
                match = display_pattern.search(modified_source_content)
            lines_before = (
                modified_source_content[: match.start()].count("\n") if match else 0
            )

            linked_filename = suggestion["filename"]
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

            processed_target_content = app.process_wikilinks_for_display(target_content)
            target_area.load_text(processed_target_content)
            app.add_wikilink_highlights(
                target_area, target_content, processed_target_content
            )

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
            yield Label(
                f"This will scan {self.note_count} notes for wikilink suggestions.",
                id="confirm-message",
            )
            yield Label(
                "This process may take several minutes depending on the number of notes.",
                id="confirm-warning",
            )
            yield Label("")
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Start Scan", id="confirm-btn", variant="primary")

    def on_button_pressed(self, event) -> None:
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        elif event.button.id == "cancel-btn":
            self.dismiss(False)


class SearchApp(App):
    CSS_PATH = "styles.tcss"

    def process_wikilinks_for_display(self, content: str) -> str:
        """Replace wikilink syntax with display text for preview rendering."""
        import re

        pattern = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

        def replace_match(match):
            link_target = match.group(1)
            display_text = match.group(2) if match.group(2) else link_target
            return display_text

        return pattern.sub(replace_match, content)

    def add_wikilink_highlights(
        self, text_area: TextArea, original_content: str, processed_content: str
    ) -> None:
        """Add syntax highlights for wikilinks in the given TextArea based on processed content."""
        import re

        pattern = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
        original_lines = original_content.split("\n")
        processed_lines = processed_content.split("\n")

        for line_idx, (orig_line, proc_line) in enumerate(
            zip(original_lines, processed_lines)
        ):
            for match in pattern.finditer(orig_line):
                # Highlight the entire [[ ]] wikilink syntax
                start_pos = match.start()
                end_pos = match.end()
                text_area._highlights[line_idx].append((start_pos, end_pos, "link.uri"))

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
        self.suggestions_cache = None
        self.test_mode = False
        self._notes_dir: Path | None = None
        self.include_subdirs: List[str] | None = None

    def compose(self) -> ComposeResult:
        return
        yield

    def on_mount(self) -> None:
        self.push_screen("loading")
        self.set_timer(0.05, self.start_async_initialization)

    def start_async_initialization(self) -> None:
        self.call_later(self.initialize_app_async)

    async def initialize_app_async(self) -> None:
        if self.test_mode:
            await self.show_main_interface()
            return

        global model, cache

        loading_screen = cast(LoadingScreen, self.screen)
        loading_screen.update_progress(0)

        try:
            args = self.parse_arguments()

            loading_screen.update_status("Initializing...")
            await self.force_ui_update()

            loading_screen.update_status("Checking GPU availability...")
            loading_screen.update_progress(5)
            await self.force_ui_update()

            from ai import get_gpu_status

            gpu_status = get_gpu_status()

            if gpu_status["available"]:
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
                raise Exception(
                    "Failed to load AI model. Please check your installation and try again."
                )
            loading_screen.update_progress(40)
            await self.force_ui_update()

            loading_screen.update_status("Scanning notes...")
            loading_screen.update_progress(50)
            await self.force_ui_update()

            current_notes = get_all_notes(self.get_notes_dir(), self.include_subdirs)
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
                model,
                current_note_paths,
                self.get_notes_dir(),
                args.rebuild_cache,
                progress_callback=progress_callback,
            )
            cache = ai_cache
            loading_screen.update_progress(95)
            loading_screen.update_status("Starting application...")
            loading_screen.update_progress(100)
            await self.force_ui_update()
            await self.show_main_interface()

        except Exception as e:
            loading_screen.update_progress(0)
            loading_screen.update_status(f"Error: {str(e)}")

    async def force_ui_update(self) -> None:
        await asyncio.sleep(0.01)

    async def show_main_interface(self) -> None:
        self.loading = False
        self.switch_screen("search")
        await self.force_ui_update()
        self.initialize_search_screen()
        self.update_mode_button()

    def initialize_search_screen(self) -> None:
        search_screen = cast(SearchScreen, self.screen)
        try:
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
        except:
            pass

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

        results = await asyncio.to_thread(
            search, query, model, cache, max_results=self.MAX_RESULTS
        )

        self.current_results = results
        self.selected_index = 0

        self.display_results()
        self.screen.refresh()

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
            results_id = (
                "#results" if self.app_mode == MODE_SEARCH else "#analyze-results"
            )
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
            title = self.get_note_title(content, str(path))
            preview = content.replace("\n", " ").strip()[:100]
            if len(content) > 100:
                preview += "..."

            is_low_relevance = score < self.SCORE_THRESHOLD

            if i == self.selected_index:
                self.write_selected_result(log, score, title, preview, is_low_relevance)
            else:
                self.write_unselected_result(
                    log, score, title, preview, is_low_relevance
                )

    def write_analysis_info(self, log, query):
        log.write(f"[dim]Analyzing text for wikilink suggestions[/dim]")
        log.write("")

        if query:
            log.write(
                f'[bold #cd5c5c]Found {len(self.current_results)} wikilink suggestions for "{query}"[/bold #cd5c5c]'
            )
        else:
            log.write(
                f"[bold #cd5c5c]Found {len(self.current_results)} wikilink suggestions[/bold #cd5c5c]"
            )
        log.write("")

    def get_note_title(self, content, filename):
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return Path(filename).stem

    def write_wikilink_suggestions(self, log):
        if hasattr(self, "current_suggestions") and self.current_suggestions:
            for i, suggestion in enumerate(self.current_suggestions):
                score = suggestion["score"]
                candidate = suggestion["candidate"]
                filename = suggestion["filename"]
                source_note = suggestion.get("source_note", "Unknown")
                target_title = self.get_note_title(
                    suggestion.get("note_content", ""), filename
                )
                is_selected = i == self.selected_index

                if is_selected:
                    log.write(
                        f"[bold #f5dede on #5d2828]{score:.3f}  {source_note} ({candidate}) -> {filename} ({target_title})[/]"
                    )
                else:
                    log.write(
                        f"[#cd5c5c]{score:.3f}  {source_note} ({candidate}) -> {filename} ({target_title})"
                    )

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
            if self.current_results and 0 <= self.selected_index < len(
                self.current_results
            ):
                score, note_path, note_content = self.current_results[
                    self.selected_index
                ]
                filename = Path(note_path).name
                results_log.write(
                    f"[bold #cd5c5c]Wikilink suggestions for: {filename}[/bold #cd5c5c]"
                )
            else:
                results_log.write("[bold #cd5c5c]Wikilink suggestions[/bold #cd5c5c]")

            results_log.write("")
            results_log.write(f"Found {len(self.current_analysis_results)} suggestions")
            results_log.write("")

            # Display wikilink suggestions
            for i, (score, filename, wikilink) in enumerate(
                self.current_analysis_results
            ):
                is_selected = i == self.selected_suggestion_index

                if is_selected:
                    results_log.write(
                        f"[bold #f5dede on #5d2828]{score:.3f}  {wikilink}[/]"
                    )
                else:
                    results_log.write(f"[#cd5c5c]{score:.3f}  {wikilink}")

                results_log.write("")

        except Exception as e:
            print(f"Error displaying analysis results: {e}")

    def display_search_preview(self, index: int, preview_area: TextArea) -> None:
        """Display preview for search mode."""
        score, path, content = self.current_results[index]
        rel_path = str(path).replace(str(Path.home()), "~")
        title = self.get_note_title(content, str(path))

        # Combine header and content (keep wikilinks for highlighting)
        header = f"{title}\n{rel_path}\n"
        header += f"Score: {score:.4f}\n\n"
        full_content = header + content

        preview_area.clear()
        preview_area.load_text(full_content)

        # Add wikilink highlights
        self.add_wikilink_highlights(preview_area, full_content, full_content)

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
                    self.selected_suggestion_index = max(
                        0, self.selected_suggestion_index - 1
                    )
                else:
                    self.selected_suggestion_index = min(
                        len(self.all_analysis_suggestions) - 1,
                        self.selected_suggestion_index + 1,
                    )

                analyze_screen = cast(AnalyzeScreen, self.screen)
                analyze_screen.display_all_analysis_results()
                analyze_screen.display_analysis_preview(self.selected_suggestion_index)

    def debounce_search(self, query):
        if hasattr(self, "_search_timer") and self._search_timer:
            self._search_timer.stop()

        async def do_search():
            await self.perform_search(query)

        self._search_timer = self.set_timer(
            0.3, lambda: asyncio.create_task(do_search())
        )

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

        else:
            self.app_mode = MODE_SEARCH
            self.switch_screen("search")

        self.call_later(self.update_mode_button)

    async def scan_all_notes_for_wikilinks(self) -> None:
        global model, cache

        if not cache or self.loading:
            return

        if self.suggestions_cache is not None:
            self.all_analysis_suggestions = self.suggestions_cache
            self.selected_suggestion_index = 0
            analyze_screen = cast(AnalyzeScreen, self.screen)
            analyze_screen.display_all_analysis_results()
            if self.suggestions_cache:
                analyze_screen.display_analysis_preview(0)
            return

        try:
            analyze_screen = cast(AnalyzeScreen, self.screen)

            total_notes = len(cache)
            analyze_screen.update_progress(
                0, f"Preparing to scan {total_notes} notes..."
            )

            all_suggestions = []
            processed_count = 0

            for note_path, (note_content, note_embedding) in cache.items():
                filename = Path(note_path).name

                progress_percent = int((processed_count / total_notes) * 100)
                analyze_screen.update_progress(
                    progress_percent, f"Analyzing: {filename}"
                )

                nlp = load_spacy_model()
                suggestions = analyze_text_for_wikilinks(
                    note_content, model, cache, nlp, note_path
                )

                for suggestion in suggestions:
                    suggestion["source_note"] = filename
                    suggestion["source_note_path"] = note_path
                    suggestion["source_note_content"] = (
                        note_content[:200] + "..."
                        if len(note_content) > 200
                        else note_content
                    )

                all_suggestions.extend(suggestions)
                processed_count += 1

                await asyncio.sleep(0.01)

            all_suggestions.sort(key=lambda x: x["score"], reverse=True)

            # Limit to top 1000 suggestions for performance
            all_suggestions = all_suggestions[:1000]

            self.suggestions_cache = all_suggestions
            self.all_analysis_suggestions = all_suggestions
            self.selected_suggestion_index = 0

            analyze_screen.update_progress(
                100, f"Scan complete! Found {len(all_suggestions)} suggestions"
            )
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
            suggestions = analyze_text_for_wikilinks(
                note_content, model, cache, nlp, note_path
            )

            self.current_analysis_results = [
                (s["score"], s["filename"], s["wikilink"]) for s in suggestions
            ]
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

        self._analysis_timer = self.set_timer(
            0.3, lambda: asyncio.create_task(do_analysis())
        )

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
            "--rebuild-cache",
            action="store_true",
            help="Force rebuild the search cache from scratch",
        )
        parser.add_argument(
            "--include-subdirs",
            help="Comma-separated list of subdirectories to include (relative to notes_dir)",
        )
        try:
            return parser.parse_args()
        except SystemExit:
            # During testing, pytest may pass unknown arguments, so return defaults
            return argparse.Namespace(
                notes_dir="test_data", rebuild_cache=False, include_subdirs=None
            )

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
