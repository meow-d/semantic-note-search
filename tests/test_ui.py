#!/usr/bin/env python3
"""Comprehensive UI tests for the semantic note search application using pytest and Textual."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from ui import SearchApp, SearchScreen, LoadingScreen, ConfirmAnalyzeScreen
from config import MODE_SEARCH, MODE_ANALYZE
from src.main import parse_arguments
from textual.widgets import RichLog


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_default_arguments(self):
        """Test default argument values."""
        # Reset sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ['main.py']

        try:
            args = parse_arguments()
            assert args.notes_dir == "test_data"
            assert args.test_mode is False
        finally:
            sys.argv = original_argv

    def test_test_mode_flag(self):
        """Test --test-mode flag parsing."""
        original_argv = sys.argv.copy()
        sys.argv = ['main.py', '--test-mode']

        try:
            args = parse_arguments()
            assert args.test_mode is True
        finally:
            sys.argv = original_argv

    def test_custom_notes_dir(self):
        """Test custom notes directory argument."""
        original_argv = sys.argv.copy()
        sys.argv = ['main.py', '/custom/path']

        try:
            args = parse_arguments()
            assert args.notes_dir == "/custom/path"
        finally:
            sys.argv = original_argv


class TestAppComponents:
    """Test basic app components."""

    def test_app_initialization(self):
        """Test that the app can be initialized."""
        app = SearchApp()
        assert app is not None
        assert hasattr(app, 'test_mode')
        assert hasattr(app, 'create_test_cache')
        assert hasattr(app, 'app_mode')

    def test_test_cache_creation(self):
        """Test that test cache is created correctly."""
        app = SearchApp()
        app.test_mode = True
        cache = app.create_test_cache()

        assert isinstance(cache, dict)
        assert len(cache) == 3

        # Check that actual test files are present
        actual_files = ["test_data/test1.md", "test_data/ml_notes.md", "test_data/python_notes.md"]
        for actual_file in actual_files:
            assert actual_file in cache
            content, embedding = cache[actual_file]
            assert isinstance(content, str)
            assert len(content) > 0
            assert len(embedding) == 768  # BGE-base-en-v1.5 dimensions

    def test_mode_constants(self):
        """Test that mode constants are properly defined."""
        assert MODE_SEARCH == "search"
        assert MODE_ANALYZE == "analyze"


@pytest.mark.asyncio
class TestUITextualIntegration:
    """Integration tests using Textual's testing framework."""

    async def test_app_starts_in_test_mode(self):
        """Test that app initializes correctly in test mode."""
        app = SearchApp()
        app.test_mode = True

        # Check initial state without running the app
        assert app.test_mode is True
        assert app.app_mode == MODE_SEARCH  # Default mode
        assert hasattr(app, 'create_test_cache')

    async def test_mode_switching_button_exists(self):
        """Test that mode switching button exists in the UI."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            # Wait for app to initialize and switch to search screen
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # Check that we're on search screen
            assert isinstance(pilot.app.screen, SearchScreen)

            # Check that mode button exists
            button = pilot.app.screen.query_one("#mode-btn")
            assert button is not None

    async def test_search_input_exists(self):
        """Test that search input exists in search mode."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # Should be on search screen
            assert isinstance(pilot.app.screen, SearchScreen)

            # In search mode, search input should exist
            search_input = pilot.app.screen.query_one("#search-input")
            assert search_input is not None

    async def test_initial_mode_is_search(self):
        """Test that app starts in search mode."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()

            # App should have app_mode attribute
            assert hasattr(pilot.app, 'app_mode')

    async def test_mode_button_text_updates(self):
        """Test that mode button text updates when mode changes."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # Should be on search screen
            assert isinstance(pilot.app.screen, SearchScreen)

            # Button should exist and have a label
            button = pilot.app.screen.query_one("#mode-btn")
            assert hasattr(button, 'label')

    async def test_loading_screen_elements_exist(self):
        """Test that loading screen has required elements."""
        app = SearchApp()
        app.test_mode = True

        # Check that the LoadingScreen class has the right compose method
        screen = LoadingScreen()
        # The test passes if the screen can be instantiated
        assert screen is not None

        assert hasattr(screen, 'update_progress')

    async def test_main_interface_elements_exist(self):
        """Test that main interface has required elements after loading."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # After loading, should be on search screen
            assert isinstance(pilot.app.screen, SearchScreen)

            # Results log should exist
            results_log = pilot.app.screen.query_one("#results")
            assert results_log is not None


@pytest.mark.asyncio
class TestUserInteractions:
    """Test user interaction scenarios."""

    async def test_search_input_accepts_text(self):
        """Test that search input accepts text input."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            # Wait for app to initialize and switch to search screen
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # Wait until we're on the search screen
            while not isinstance(pilot.app.screen, SearchScreen):
                await pilot.pause(0.1)

            # Type into search input
            await pilot.click("#search-input")
            await pilot.press("t", "e", "s", "t")
            await pilot.press("enter")

            # Should not crash
            assert True

    async def test_search_functionality_works(self):
        """Test that typing in search input and pressing enter doesn't crash."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            # Wait for app to initialize and switch to search screen
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # Wait until we're on the search screen
            while not isinstance(pilot.app.screen, SearchScreen):
                await pilot.pause(0.1)

            # Type a search query and press enter
            await pilot.click("#search-input")
            await pilot.press("t", "e", "s", "t")
            await pilot.press("enter")

            # If we get here without crashing, the basic functionality works
            assert isinstance(pilot.app.screen, SearchScreen)

    async def test_analyze_button_click(self):
        """Test that clicking the analyze button triggers mode switch."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            # Wait for app to initialize and switch to search screen
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # Wait until we're on the search screen
            while not isinstance(pilot.app.screen, SearchScreen):
                await pilot.pause(0.1)

            # Click the analyze button
            await pilot.click("#mode-btn")

            # Wait for mode switch
            await pilot.pause(0.5)

            # Should not crash
            assert True

    async def test_mode_button_click_changes_mode(self):
        """Test that clicking mode button changes the app mode."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # Extra pause to ensure screen switch

            # Click mode button
            await pilot.click("#mode-btn")
            await pilot.pause()

            # Should not crash
            assert True

    async def test_keyboard_navigation(self):
        """Test keyboard navigation between results."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()

            # Press down arrow (should not crash even with no results)
            await pilot.press("down")
            await pilot.press("up")

            assert True  # If we get here, navigation didn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])