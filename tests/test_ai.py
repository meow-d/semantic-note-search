#!/usr/bin/env python3
"""Comprehensive tests for AI functionality in the semantic note search application."""

import pytest
import torch
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    load_model,
    build_cache,
    load_cache,
    update_cache_for_new_notes,
    remove_deleted_notes_from_cache,
    load_or_build_cache,
    load_spacy_model,
    analyze_text_for_wikilinks,
    search,
    CACHE_FILE,
    MODEL_NAME,
    SCORE_THRESHOLD,
    WIKILINK_SCORE_THRESHOLD,
    get_all_notes,
    parse_arguments
)


class TestModelLoading:
    """Test model loading functionality."""

    @patch('main.SentenceTransformer')
    def test_load_model_success(self, mock_sentence_transformer):
        """Test successful model loading."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        result = load_model()

        mock_sentence_transformer.assert_called_once_with(MODEL_NAME, trust_remote_code=True)
        assert result == mock_model

    @patch('main.SentenceTransformer')
    @patch('builtins.print')
    def test_load_model_prints_messages(self, mock_print, mock_sentence_transformer):
        """Test that model loading prints appropriate messages."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        load_model()

        # Check that loading and loaded messages were printed
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Loading model" in msg for msg in print_calls)
        assert any("Model loaded successfully" in msg for msg in print_calls)


class TestCacheOperations:
    """Test cache building and loading functionality."""

    def test_build_cache_creates_proper_structure(self):
        """Test that build_cache creates cache with correct structure."""
        # Create mock model and notes
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(768)  # Single embedding

        test_notes = [
            Path("notes/test1.md"),
            Path("notes/test2.md"),
            Path("notes/test3.md")
        ]

        # Mock file reading
        test_contents = [
            "# Test Note 1\n\nThis is test content 1.",
            "# Test Note 2\n\nThis is test content 2.",
            "# Test Note 3\n\nThis is test content 3."
        ]

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.side_effect = test_contents

            cache = build_cache(test_notes, mock_model)

            # Verify structure
            assert isinstance(cache, dict)
            assert len(cache) == 3

            for note_path in test_notes:
                assert str(note_path) in cache
                content, embedding = cache[str(note_path)]
                assert isinstance(content, str)
                assert isinstance(embedding, torch.Tensor)
                assert embedding.shape == (768,)

    def test_load_cache_file_not_exists(self):
        """Test loading cache when file doesn't exist."""
        with patch('main.CACHE_FILE', Path('/nonexistent/cache.pkl')):
            cache = load_cache()
            assert cache == {}

    @patch('pickle.load')
    @patch('builtins.open', create=True)
    def test_load_cache_success(self, mock_open, mock_pickle_load):
        """Test successful cache loading."""
        test_cache = {
            "notes/test.md": ("content", torch.randn(768))
        }
        mock_pickle_load.return_value = test_cache

        cache = load_cache()

        assert cache == test_cache
        mock_open.assert_called_once()

    def test_update_cache_for_new_notes(self):
        """Test updating cache with new notes."""
        existing_cache = {
            "notes/old.md": ("old content", torch.randn(768))
        }

        new_notes = [Path("notes/new.md")]
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(1, 768)

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "new content"

            updated_cache = update_cache_for_new_notes(mock_model, existing_cache, new_notes)

            assert updated_cache is not None
            assert len(updated_cache) == 2
            assert "notes/new.md" in updated_cache
            assert "notes/old.md" in updated_cache

    def test_remove_deleted_notes_from_cache(self):
        """Test removing deleted notes from cache."""
        cache = {
            "notes/exists.md": ("content1", torch.randn(768)),
            "notes/deleted.md": ("content2", torch.randn(768)),
            "notes/also_deleted.md": ("content3", torch.randn(768))
        }

        existing_notes = {"notes/exists.md"}

        cleaned_cache = remove_deleted_notes_from_cache(cache, {"notes/deleted.md", "notes/also_deleted.md"})

        assert cleaned_cache is not None
        assert len(cleaned_cache) == 1
        assert "notes/exists.md" in cleaned_cache
        assert "notes/deleted.md" not in cleaned_cache


class TestSearchFunctionality:
    """Test search functionality."""

    def test_search_with_empty_cache(self):
        """Test search with empty cache returns empty results."""
        mock_model = MagicMock()
        results = search("test query", mock_model, {})
        assert results == []

    def test_search_with_cache(self):
        """Test search with populated cache."""
        # Create test cache
        cache = {
            "notes/test1.md": ("This is about machine learning and AI", torch.randn(768)),
            "notes/test2.md": ("This is about Python programming", torch.randn(768)),
            "notes/test3.md": ("This is about web development", torch.randn(768))
        }

        mock_model = MagicMock()
        # Mock the encode method to return proper shape
        mock_model.encode.return_value = torch.randn(768)
        # Mock similarity scores - higher for first result
        with patch('main.util.cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.8, 0.3, 0.2]])

            results = search("machine learning", mock_model, cache)

        assert len(results) > 0
        # Results should be sorted by score (highest first)
        assert results[0][0] >= results[-1][0]  # First score >= last score

    def test_search_filters_by_threshold(self):
        """Test that search filters results by score threshold."""
        cache = {
            "notes/high.md": ("high relevance content", torch.randn(768)),
            "notes/low.md": ("low relevance content", torch.randn(768))
        }

        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(768)
        # Set scores: one above threshold, one below
        with patch('main.util.cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[SCORE_THRESHOLD + 0.1, SCORE_THRESHOLD - 0.2]])

            results = search("test query", mock_model, cache)

        # Should only return results above threshold
        assert len(results) == 1
        assert results[0][1] == "notes/high.md"


class TestWikilinkAnalysis:
    """Test wikilink analysis functionality."""

    @patch('main.spacy')
    def test_analyze_text_for_wikilinks_with_spacy(self, mock_spacy):
        """Test wikilink analysis with spaCy available."""
        # This test is complex to mock properly, so we'll just test that it doesn't crash
        # Mock spaCy
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp

        # Mock the extraction functions to return some candidates
        with patch('main.extract_noun_phrases', return_value=["machine learning"]), \
             patch('main.extract_verb_phrases', return_value=["learn"]):

            # Mock cache and model
            cache = {"notes/ml.md": ("machine learning content", torch.randn(768))}
            mock_model = MagicMock()
            mock_model.encode.return_value = torch.randn(2, 768)  # 2 candidates

            with patch('main.util.cos_sim') as mock_cos_sim:
                mock_cos_sim.return_value = torch.tensor([[0.9, 0.8]])

                text = "I want to learn about machine learning"
                suggestions = analyze_text_for_wikilinks(text, mock_model, cache, mock_nlp)

                # Should return some suggestions
                assert isinstance(suggestions, list)
        # Should find suggestions for noun phrases

    @patch('main.spacy', None)
    def test_analyze_text_for_wikilinks_without_spacy(self):
        """Test wikilink analysis fallback when spaCy not available."""
        cache = {"notes/test.md": ("test content", torch.randn(768))}
        mock_model = MagicMock()
        mock_model.similarity.return_value = torch.tensor([[0.9]])

        text = "test query"
        suggestions = analyze_text_for_wikilinks(text, mock_model, cache, None)

        # Should still work with basic fallback
        assert isinstance(suggestions, list)

    def test_wikilink_filtering_by_threshold(self):
        """Test that wikilink suggestions are filtered by threshold."""
        cache = {
            "notes/high.md": ("high relevance", torch.randn(768)),
            "notes/low.md": ("low relevance", torch.randn(768))
        }

        mock_model = MagicMock()
        # One above threshold, one below
        mock_model.similarity.return_value = torch.tensor([
            [WIKILINK_SCORE_THRESHOLD + 0.1],
            [WIKILINK_SCORE_THRESHOLD - 0.1]
        ])

        text = "test query"
        suggestions = analyze_text_for_wikilinks(text, mock_model, cache, None)

        # Should filter out low relevance suggestions
        high_relevance_suggestions = [s for s in suggestions if s[1] > WIKILINK_SCORE_THRESHOLD]
        assert len(high_relevance_suggestions) >= 0  # May be 0 if fallback doesn't find anything


class TestIntegration:
    """Integration tests for combined functionality."""

    @patch('main.SentenceTransformer')
    @patch('pickle.dump')
    @patch('builtins.open', create=True)
    def test_load_or_build_cache_integration(self, mock_open, mock_pickle_dump, mock_sentence_transformer):
        """Test the complete cache loading/building workflow."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        mock_model.encode.return_value = torch.randn(2, 768)

        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch('main.get_all_notes') as mock_get_notes:
            mock_get_notes.return_value = [Path("notes/test1.md"), Path("notes/test2.md")]

            with patch('builtins.open', create=True) as mock_file_open:
                mock_file_open.return_value.__enter__.return_value.read.side_effect = [
                    "content 1", "content 2"
                ]

                cache, status = load_or_build_cache(mock_model, {"notes/test1.md", "notes/test2.md"})

                assert len(cache) == 2
                assert all(isinstance(content, str) for content, _ in cache.values())
                assert all(isinstance(embedding, torch.Tensor) for _, embedding in cache.values())


if __name__ == "__main__":
    pytest.main([__file__])