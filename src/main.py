#!/usr/bin/env python3

"""Main entry point for the semantic note search application."""

import argparse
import sys
from pathlib import Path

from .ui import SearchApp


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
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild the search cache from scratch",
    )
    parser.add_argument(
        "--include-subdirs",
        help="Comma-separated list of subdirectories to include (relative to notes_dir)",
    )
    try:
        return parser.parse_args()
    except SystemExit:
        # During testing, pytest may pass unknown arguments, so return defaults
        return argparse.Namespace(notes_dir="test_data", rebuild_cache=False)


def main():
    """Main application entry point."""
    args = parse_arguments()
    notes_dir = Path(args.notes_dir)
    include_subdirs = args.include_subdirs.split(",") if args.include_subdirs else None

    print(f"Using notes directory: {notes_dir}")
    if include_subdirs:
        print(f"Including subdirectories: {include_subdirs}")

    if not notes_dir.is_dir():
        print(f"Error: Notes directory not found at '{notes_dir}'")
        print(
            f"Create the directory or specify a different one: python main.py /path/to/notes"
        )
        sys.exit(1)

    app = SearchApp()
    app.notes_dir = notes_dir  # Pass the validated notes directory
    app.include_subdirs = include_subdirs
    app.run()


if __name__ == "__main__":
    main()
