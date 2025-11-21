"""Configuration constants for the semantic note search application."""

from pathlib import Path

# Model and AI constants
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SCORE_THRESHOLD = 0.52
WIKILINK_SCORE_THRESHOLD = 0.65
MAX_RESULTS = 16

# Application modes
MODE_SEARCH = "search"
MODE_ANALYZE = "analyze"


def get_cache_file(notes_dir):
    """Get cache file path based on notes directory."""
    # Use notes directory name for cache file, store in project root
    notes_name = Path(notes_dir).name
    cache_name = f"cache_{notes_name}.pkl"
    # Store in project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    return project_root / cache_name
