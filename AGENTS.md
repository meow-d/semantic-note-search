# Agent Guidelines for Semantic Note Search

## Commands
- **Run**: `python main.py`
- **Dependencies**: `pip install textual sentence-transformers torch "textual[syntax]"`
- **Test**: No formal tests yet - test manually by running the app
- **Single test**: No test framework configured

## Code Style
- **Imports**: Group imports: stdlib → third-party → local, use absolute imports
- **Formatting**: Follow PEP 8, 4-space indentation, 79-char line limit
- **Types**: Use type hints where helpful, especially for function signatures
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use try/except for file operations, provide clear error messages
- **Documentation**: Use docstrings for public functions and classes
- **Architecture**: Single-file application, keep concerns separated (UI vs business logic)
- Stop importing static and label if you're not gonna use it

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

## Issues
- see README.md. do NOT edit the todo there except for ticking.

- for the agent's own todo use below:

## Agent TODO

