# Agent Guidelines for Semantic Note Search

## Commands
- **Run**: `python main.py`
- **Run in test mode**: `python main.py --test-mode`
- **Dependencies**: `pip install textual sentence-transformers torch "textual[syntax]" pytest pytest-asyncio pytest-textual-snapshot`
- **Test**: `pytest` (comprehensive test suite)
- **Single test**: `pytest tests/test_ai.py::test_search_functionality`

## Code Style
- **Imports**: Group imports: stdlib → third-party → local, use absolute imports
- **Formatting**: Follow PEP 8, 4-space indentation, 79-char line limit
- **Types**: Use type hints where helpful, especially for function signatures
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use try/except for file operations, provide clear error messages
- **Documentation**: Use docstrings for public functions and classes
- **Architecture**: Single-file application, keep concerns separated (UI vs business logic)

## Project Structure
- **Single Python file**: All functionality in `main.py`
- **Configuration**: Constants at top, CLI/TUI app class at bottom
- **Dependencies**: textual for TUI, sentence-transformers for ML, torch for tensor ops

## Key Requirements
- Interactive TUI with search bar at bottom, results left, preview right
- Arrow key navigation between results
- Real-time search as you type
- Persistent caching with pickle for performance
- Handle missing dependencies gracefully

## Exit Commands
- Press `q`, `escape`, or `ctrl+c` to quit

## Other
- Try to use search MCP tools to search for documentation
- The user may edit the files. Do NOT override the user's changes.

## Critical Notes
- **DO NOT touch the user's cache** (`cache.pkl`). The cache can take hours to build depending on the size of the note collection and the model used. Never rebuild, delete, or modify the user's existing cache file. Only create new caches for testing purposes in isolated environments.

## Issues
- See README.md for current issues and TODO items

