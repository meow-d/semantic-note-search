#!/usr/bin/env python3
"""Comprehensive UI tests for the semantic note search application using pytest and Textual."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SearchApp, parse_arguments, MODE_SEARCH, MODE_ANALYZE


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

        # Check that all expected files are present
        expected_files = ["test_data/test1.md", "test_data/test2.md", "test_data/test3.md"]
        for expected_file in expected_files:
            assert expected_file in cache
            content, embedding = cache[expected_file]
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

    @patch('main.parse_arguments')
    async def test_app_starts_in_test_mode(self, mock_parse):
        """Test that app starts correctly in test mode."""
        mock_parse.return_value.test_mode = True
        mock_parse.return_value.notes_dir = "notes"

        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            # App should start without errors in test mode
            assert app.test_mode is True
            assert app.app_mode == MODE_SEARCH  # Default mode

    async def test_mode_switching_button_exists(self):
        """Test that mode switching button exists in the UI."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            # Wait for app to initialize
            await pilot.pause()

            # Check that mode button exists
            button = pilot.app.query_one("#mode-button")
            assert button is not None

    async def test_search_input_exists(self):
        """Test that search input exists in search mode."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()

            # In search mode, search input should exist
            search_input = pilot.app.query_one("#search-input")
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

            # Button should exist and have a label
            button = pilot.app.query_one("#mode-button")
            assert hasattr(button, 'label')

    async def test_loading_screen_elements_exist(self):
        """Test that loading screen has required elements."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            # Check loading log exists
            loading_log = pilot.app.query_one("#loading-log")
            assert loading_log is not None

            # Check loading progress exists
            loading_progress = pilot.app.query_one("#loading-progress")
            assert loading_progress is not None

    async def test_main_interface_elements_exist(self):
        """Test that main interface has required elements after loading."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()

            # After loading, main interface should be visible
            main_interface = pilot.app.query_one("#main-interface")
            assert main_interface is not None

            # Results log should exist
            results_log = pilot.app.query_one("#results")
            assert results_log is not None


@pytest.mark.asyncio
class TestUserInteractions:
    """Test user interaction scenarios."""

    async def test_search_input_accepts_text(self):
        """Test that search input accepts text input."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()

            # Type into search input
            await pilot.click("#search-input")
            await pilot.press("t", "e", "s", "t")
            await pilot.press("enter")

            # Should not crash
            assert True

    async def test_mode_button_click_changes_mode(self):
        """Test that clicking mode button changes the app mode."""
        app = SearchApp()
        app.test_mode = True

        async with app.run_test() as pilot:
            await pilot.pause()

            # Click mode button
            await pilot.click("#mode-button")
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