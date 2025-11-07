#!/usr/bin/env python3

"""Main entry point for the semantic note search application."""

import argparse
import sys
from pathlib import Path

from ui import SearchApp


def parse_arguments():
    """Parse command line arguments."""
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
        "--test-mode",
        action="store_true",
        help="Run in test mode with dummy data (no AI loading)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild the search cache from scratch",
    )
    try:
        return parser.parse_args()
    except SystemExit:
        # During testing, pytest may pass unknown arguments, so return defaults
        return argparse.Namespace(notes_dir="test_data", test_mode=False, rebuild_cache=False)


def main():
    """Main application entry point."""
    args = parse_arguments()
    notes_dir = Path(args.notes_dir)

    print(f"Using notes directory: {notes_dir}")

    if not notes_dir.is_dir():
        print(f"Error: Notes directory not found at '{notes_dir}'")
        print(
            f"Create the directory or specify a different one: python main.py /path/to/notes"
        )
        sys.exit(1)

    app = SearchApp()
    app.test_mode = args.test_mode  # Set test mode from arguments
    app.run()


if __name__ == "__main__":
    main()
