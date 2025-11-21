# Agent Guidelines for Semantic Note Search
1. Look at the README.md TODOs for your list of current tasks. do not touch the readme TODOs except for ticking the checkmarks.
2. select a task to do. for complex tasks, break them down into subtasks. rule of thumb: ensure the app is always in a working state at all times. do NOT pull or push, you must only use git status, add, and commit.
3. make sure all tests pass and everything is working
4. make a git commit
5. moving on to the next task, repeat

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
- Comments: Document with docstrings whenever possible. Don't comment everywhere else unless it's nessasary.
- **Imports**: Group imports: stdlib → third-party → local, use absolute imports
- **Formatting**: Follow PEP 8, 4-space indentation, 79-char line limit
- **Types**: Always use type hints when possible
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: use try catch for file operation. the app must be able to tolerate failures. however, it should not fail silently, not in ways that the user cannot tell something is wrong.
- **Documentation**: Use docstrings for public functions and classes
- **Architecture**: keep concerns separated (UI vs business logic)
- **Commits**: Use semantic commits, like `feat: implement caching through pickle files`

## Project Structure
- **Dependencies**: textual for TUI, sentence-transformers for ML, torch for tensor ops

## Other
- ALways use the search MCP tools to search for documentation on the internet
- The user may edit the files. Do NOT override the user's changes.
- **DO NOT touch the user's cache** (`cache.pkl`). The cache can take hours to build depending on the size of the note collection and the model used. Never rebuild, delete, or modify the user's existing cache file. Only create new caches for testing purposes in isolated environments.

