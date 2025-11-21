# Semantic Note Search
### Unserious introduction
🚀 Meet Semantic Note Search ⚡ — where your ideas finally talk back.

We’re redefining note-taking for the AI era. This isn’t another productivity app — it’s your thinking partner, your memory amplifier, your second brain’s interface layer. Ask anything in plain language, and Semantic Note Search finds, connects, and contextualizes your notes like it already knew what you meant.

Forget the old days of keyword hunting and file spelunking — that’s peasant tech. We built Semantic Note Search to make your notes feel alive, to surface insights you didn’t even remember writing.

🧠 Backed by cutting-edge semantic models and obsessive design, this is note search that understands you.
Not just faster — smarter. Personal. Effortless.

Welcome to the future of knowledge flow.
Welcome to Semantic Note Search — your thoughts, instantly searchable. ✨

### Serious introduction
Semantic search TUI for markdown (and .txt) notes using Sentence Transformers. vibe coded, for personal use when i was bored this afternoon. unfortunately i don't actually have a use for it, because i link my notes properly i never had trouble finding notes in the first place.

## Usage
```bash
# install
git clone meow-d/semantic-note-search
cd semantic-note-search
pip install -r requirements.txt

# run
python main.py /path/to/your/notes
```

First run will take a while as it'll need to download the sentence transformer model and build a search index. Subsequent runs might also take a while depending on the dataset size.

## Requirements
- Python 3.8+
- Textual
- Sentence Transformers
- PyTorch
- [textual[syntax]] (for markdown highlighting)

## License
MIT License - feel free to use, modify, and distribute.

## Todo
- [x] in the wikilink previews, top right, wikilinks are not applied. right now it's "This is a test note about artificial intelligence and machine learning.". it should be "This is a test note about [[ml_notes|artificial intelligence]] and machine learning."
- [x] look into textual tui documentation on how to make text inside textarea highlighted. then, highllight the wikilink previews, everything between [[]]

- [x] separate the textual css in separate .tcss file
- [x] search ui should parse the results for the first h1 to use as title
- [x] search ui removed [[ and ]] in wikilinks and highlights it. i only need the highlight, not removal
- [x] remove test mode entirely
- [x] update tests
- [x] refactor: look through the codebase for any unfinished code or errors, finish or remove unused code
- [x] are wikilink suggestions cached? if not, implement that feature
- [ ] wikilink suggestions: skip urls (https://), existing wikilinks, and headings (line starts with #)
- [ ] wikilink suggestions: markdown lists starts with `- `, don't include them. no `- anki`, just `anki`
- [ ] performance isues with wikilink suggestions page with >10000 suggestions

### skip for now
- [ ] type errors in ai.py
- [ ] ui elements isn't aligned (def not something an llm agent can fix)
- [ ] wikilink suggestion: implement a button and shortcut to apply suggestion. shortcut: "ctrl+enter". button: "apply selected suggestion", located in the same section as the progress bar

