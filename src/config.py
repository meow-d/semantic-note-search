"""Configuration constants for the semantic note search application."""

from pathlib import Path

# Model and AI constants
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SCORE_THRESHOLD = 0.52
WIKILINK_SCORE_THRESHOLD = 0.65
MIN_RESULTS = 12
MAX_RESULTS = 16

# File handling
NOTE_EXTENSIONS = {".txt", ".md"}

# Application modes
MODE_SEARCH = "search"
MODE_ANALYZE = "analyze"

def get_cache_file(notes_dir):
    """Get cache file path based on notes directory."""
    import hashlib
    # Create a hash of the notes directory path for unique cache files
    dir_hash = hashlib.md5(str(notes_dir).encode()).hexdigest()[:8]
    cache_name = f"cache_{dir_hash}.pkl"
    return Path(__file__).parent / cache_name