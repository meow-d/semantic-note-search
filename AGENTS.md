# Agent Guidelines for Semantic Note Search
Look at the README.md TODOs for your list of current tasks. plan things out on your own, break tasks down, and do them one by one. rule of thumb is ensure te app is always in a working state at all times. do not touch the readme TODOs except for ticking the checkmarks.

## Commands
DO NOT RUN WITHOUT A FAILSAFE (like a 20s timer), THE TOOL DOES NOT SUPPORT TUIS
- **Run**: `python main.py`
- **Run in test mode**: `python main.py --test-mode`
- **Test**: `pytest`
- **Single test**: `pytest tests/test_ai.py::test_search_functionality`

## Code Style
- !!! Priotise human readibility above everything else !!!
- Simplicity is key, make the simplest possible solution unless you're explictly told to do something more complex
- Keep nesting below 3-4 levels
- If a class/function/file gets too complex, split them out.
- Comments: Don't comment unless it's nessasary. remember that we already have docstrings
- **Imports**: Group imports: stdlib → third-party → local, use absolute imports
- **Formatting**: Follow PEP 8, 4-space indentation, 79-char line limit
- **Types**: Always use type hints when possible
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use try/except for file operations, provide clear error messages
- **Documentation**: Use docstrings for public functions and classes
- **Architecture**: keep concerns separated (UI vs business logic)

## Project Structure
- **Dependencies**: textual for TUI, sentence-transformers for ML, torch for tensor ops

## Other
- ALways use the search MCP tools to search for documentation on the internet
- The user may edit the files. Do NOT override the user's changes.

## Critical Notes
- **DO NOT touch the user's cache** (`cache.pkl`). The cache can take hours to build depending on the size of the note collection and the model used. Never rebuild, delete, or modify the user's existing cache file. Only create new caches for testing purposes in isolated environments.

## Issues
- See README.md for current issues and TODO items

